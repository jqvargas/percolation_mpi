// -*- mode: C++; -*-
//
// Copyright (C) 2025, Rupert Nash, The University of Edinburgh.
//
// All rights reserved.
//
// This file is provided to you to complete an assessment and for
// subsequent private study. It may not be shared and, in particular,
// may not be posted on the internet. Sharing this or any modified
// version may constitute academic misconduct under the University's
// regulations.

#include "perc_gpu.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <omp.h>

#include "decomp.h"

constexpr int printfreq = 100;

// Perform a single step of the algorithm.
//
// For each point (if fluid), set it to the maximum of itself and the
// four von Neumann neighbours.
//
// Returns the total number of changed cells.
static int percolate_gpu_step(int M, int N, int const* state, int* next) {
  int nchange = 0;
  int const stride = N + 2;

  // Enhanced GPU kernel
  #pragma omp target teams distribute parallel for collapse(2) \
    map(tofrom: nchange) reduction(+:nchange)
  for (int i = 1; i <= M; ++i) {
    for (int j = 1; j <= N; ++j) {
      int const idx = i*stride + j;
      int const oldval = state[idx];
      int newval = oldval;

      // 0 => solid, so do nothing
      if (oldval != 0) {
        // Set next[i][j] to be the maximum value of state[i][j] and
        // its four nearest neighbours
        newval = std::max(newval, state[idx - stride]); // North
        newval = std::max(newval, state[idx + stride]); // South
        newval = std::max(newval, state[idx - 1]);      // West
        newval = std::max(newval, state[idx + 1]);      // East

        if (newval != oldval) {
          ++nchange;
        }
      }

      next[idx] = newval;
    }
  }
  return nchange;
}


struct GpuRunner::Impl {
  // Here you can store any parameters or data needed for
  // implementation of your version.

  ParallelDecomp p;

  // Device management
  int device_id; // Assigned GPU device ID
  int node_id;   // Node ID
  char hostname[MPI_MAX_PROCESSOR_NAME]; // Hostname for better debugging

  // Use managed memory for efficient host-device transfers
  int* state;
  int* tmp;

  // Neighbouring ranks. -1 => no neighbour
  // Ordering is NESW
  std::array<int, 4> neigh_ranks;
  std::array<int, 4> neigh_counts;

  // Halo exchange buffers
  std::vector<int> halo_send_buf;
  std::vector<int> halo_recv_buf;

  // MPI request arrays for non-blocking communication
  std::array<MPI_Request, 8> requests;  // 4 sends + 4 receives

  Impl(ParallelDecomp const& pd) : p(pd) {
    // Get hostname for better debugging
    int namelen;
    MPI_Get_processor_name(hostname, &namelen);
    
    // Set up device assignment using MPI_Comm_split_type
    MPI_Comm node_comm;
    int node_rank;
    
    // Split communicator by node
    MPI_Comm_split_type(p.comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &node_rank);
    
    // Get node ID (this requires all processes on a node to have consecutive ranks)
    int node_size;
    MPI_Comm_size(node_comm, &node_size);
    node_id = p.rank / node_size;
    
    // Get number of devices on this node
    int num_devices = 0;
    #pragma omp parallel
    {
      #pragma omp single
      {
        num_devices = omp_get_num_devices();
      }
    }
    
    // Assign device (round-robin within each node)
    if (num_devices > 0) {
      device_id = node_rank % num_devices;
    } else {
      device_id = 0; // Fallback to device 0 if no devices found
    }
    
    // Print assignment for debugging
    printf("Rank %d on node %d (hostname: %s) assigned to GPU device %d of %d available devices\n", 
           p.rank, node_id, hostname, device_id, num_devices);
    
    // Set default device
    omp_set_default_device(device_id);
    
    // Free the communicator
    MPI_Comm_free(&node_comm);

    // Allocate memory - using persistent target data constructs
    state = new int[local_size()];
    tmp = new int[local_size()];
    
    // Initialize device memory
    #pragma omp target enter data map(alloc: state[0:local_size()], tmp[0:local_size()])

    // Neighbour ranks: NESW
    // N: (pi, pj + 1)
    neigh_ranks[0] = (p.pj + 1 < p.py) ? p.global_rank(p.pi, p.pj + 1) : -1;
    neigh_counts[0] = p.sub_nx;
    // E: (pi + 1, pj)
    neigh_ranks[1] = (p.pi + 1 < p.px) ? p.global_rank(p.pi + 1, p.pj) : -1;
    neigh_counts[1] = p.sub_ny;
    // S: (pi, pj - 1)
    neigh_ranks[2] = (p.pj - 1 >= 0) ? p.global_rank(p.pi, p.pj - 1) : -1;
    neigh_counts[2] = p.sub_nx;
    // W: (pi - 1, pj)
    neigh_ranks[3] = (p.pi - 1 >= 0) ? p.global_rank(p.pi - 1, p.pj) : -1;
    neigh_counts[3] = p.sub_ny;

    // Allocate halo buffers
    auto halo_buf_size = 2 * (p.sub_nx + p.sub_ny);
    halo_send_buf.resize(halo_buf_size);
    halo_recv_buf.resize(halo_buf_size);

    // Initialize requests array
    for (auto& req : requests) {
      req = MPI_REQUEST_NULL;
    }
  }
    
  ~Impl() {
    // Clean up device memory
    #pragma omp target exit data map(delete: state[0:local_size()], tmp[0:local_size()])
    delete[] state;
    delete[] tmp;
  }

  int local_size() const {
    return (p.sub_nx + 2)*(p.sub_ny + 2);
  }

  // Communicate the valid edge sites to neighbours that exist. Fill
  // halos on the receiving side using non-blocking communication
  void halo_exchange(int* data) {
    auto const stride = p.sub_ny + 2;
    
    // Copy boundary data from device to host for sending
    #pragma omp target update from(data[0:local_size()])

    // Pack the send buffers (NESW)
    int offset = 0;
    // N
    for (int i = 1; i <= p.sub_nx; ++i, ++offset) {
      halo_send_buf[offset] = data[i * stride + p.sub_ny];
    }
    // E
    for (int j = 1; j <= p.sub_ny; ++j, ++offset) {
      halo_send_buf[offset] = data[p.sub_nx * stride + j];
    }
    // S
    for (int i = 1; i <= p.sub_nx; ++i, ++offset) {
      halo_send_buf[offset] = data[i * stride + 1];
    }
    // W
    for (int j = 1; j <= p.sub_ny; ++j, ++offset) {
      halo_send_buf[offset] = data[1*stride + j];
    }

    // Post non-blocking receives first
    offset = 0;
    for (int b = 0; b < 4; ++b) {
      if (neigh_ranks[b] != -1) {
        MPI_Irecv(halo_recv_buf.data() + offset, neigh_counts[b], MPI_INT,
                 neigh_ranks[b], 0, p.comm, &requests[b]);
      } else {
        requests[b] = MPI_REQUEST_NULL;
      }
      offset += neigh_counts[b];
    }
    
    // Post non-blocking sends
    offset = 0;
    for (int b = 0; b < 4; ++b) {
      if (neigh_ranks[b] != -1) {
        MPI_Isend(halo_send_buf.data() + offset, neigh_counts[b], MPI_INT,
                 neigh_ranks[b], 0, p.comm, &requests[b+4]);
      } else {
        requests[b+4] = MPI_REQUEST_NULL;
      }
      offset += neigh_counts[b];
    }

    // Wait for all communication to complete
    MPI_Waitall(8, requests.data(), MPI_STATUSES_IGNORE);

    // Unpack received data (NESW)
    offset = 0;
    // N
    for (int i = 1; i <= p.sub_nx; ++i, ++offset) {
      data[i * stride + p.sub_ny + 1] = halo_recv_buf[offset];
    }
    // E
    for (int j = 1; j <= p.sub_ny; ++j, ++offset) {
      data[(p.sub_nx + 1) * stride + j] = halo_recv_buf[offset];
    }
    // S
    for (int i = 1; i <= p.sub_nx; ++i, ++offset) {
      data[i * stride + 0] = halo_recv_buf[offset];
    }
    // W
    for (int j = 1; j <= p.sub_ny; ++j, ++offset) {
      data[0*stride + j] = halo_recv_buf[offset];
    }

    // Update device with new halo data
    #pragma omp target update to(data[0:local_size()])
  }
};

GpuRunner::GpuRunner(ParallelDecomp const& p) : m_impl(std::make_unique<Impl>(p)) {
}

GpuRunner::~GpuRunner() = default;

// Give each process a sub-array of size (sub_nx+2) x (sub_ny+2)
void GpuRunner::copy_in(int const* source) {
  // Copy data from host to our buffer
  std::copy(source, source + m_impl->local_size(), m_impl->state);
  
  // Transfer data to device
  #pragma omp target update to(m_impl->state[0:m_impl->local_size()])
}

// Give each process a sub-array of size (sub_nx+2) x (sub_ny+2)
void GpuRunner::copy_out(int* dest) const {
  // Transfer final results from device to host
  #pragma omp target update from(m_impl->state[0:m_impl->local_size()])
  std::copy(m_impl->state, m_impl->state + m_impl->local_size(), dest);
}

// Iteratively perform percolation of the non-zero elements until no
// changes or 4 *max(M, N) iterations.
void GpuRunner::run() {
  auto& p = m_impl->p;
  int const M = p.nx;
  int const N = p.ny;
  
  // Use persistent device data mapping for improved performance
  #pragma omp target data map(to:m_impl->state[0:m_impl->local_size()], m_impl->tmp[0:m_impl->local_size()]) \
                         map(from:m_impl->state[0:m_impl->local_size()])
  {
    // Initialize tmp on device
    #pragma omp target teams distribute parallel for collapse(2)
    for (int i = 0; i < p.sub_nx + 2; ++i) {
      for (int j = 0; j < p.sub_ny + 2; ++j) {
        m_impl->tmp[i*(p.sub_ny+2) + j] = m_impl->state[i*(p.sub_ny+2) + j];
      }
    }

    int const maxstep = 4 * std::max(M, N);
    int step = 1;
    int global_nchange = 1;

    // Use pointers to the buffers which we swap to avoid copies
    int* current = m_impl->state;
    int* next = m_impl->tmp;

    while (global_nchange && step <= maxstep) {
      // Exchange halos with neighboring processes
      m_impl->halo_exchange(current);
      
      // Run the computation kernel
      int local_nchange = percolate_gpu_step(p.sub_nx, p.sub_ny, current, next);

      // Share the total changes
      MPI_Allreduce(&local_nchange,
                   &global_nchange,
                   1, MPI_INT, MPI_SUM, p.comm);

      if (step % printfreq == 0 && p.rank == 0) {
        p.print_root("percolate: number of changes on step {} is {}\n",
                    step, global_nchange);
      }

      // Swap the pointers for the next iteration
      std::swap(next, current);
      step++;
    }

    // Ensure final state is in m_impl->state
    if (current != m_impl->state) {
      #pragma omp target teams distribute parallel for collapse(2)
      for (int i = 0; i < p.sub_nx + 2; ++i) {
        for (int j = 0; j < p.sub_ny + 2; ++j) {
          m_impl->state[i*(p.sub_ny+2) + j] = current[i*(p.sub_ny+2) + j];
        }
      }
    }
  } // End of target data region
}