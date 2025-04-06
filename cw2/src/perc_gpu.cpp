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

  // Simple GPU offloading for single device
  #pragma omp target teams distribute parallel for collapse(2) \
    map(tofrom: nchange) map(to: state[0:(M+2)*(N+2)]) map(from: next[0:(M+2)*(N+2)]) \
    reduction(+:nchange)
  for (int i = 1; i <= M; ++i) {
    for (int j = 1; j <= N; ++j) {
      int const oldval = state[i*stride + j];
      int newval = oldval;

      // 0 => solid, so do nothing
      if (oldval != 0) {
	// Set next[i][j] to be the maximum value of state[i][j] and
	// its four nearest neighbours
	newval = std::max(newval, state[(i-1)*stride + j  ]);
	newval = std::max(newval, state[(i+1)*stride + j  ]);
	newval = std::max(newval, state[    i*stride + j-1]);
	newval = std::max(newval, state[    i*stride + j+1]);

	if (newval != oldval) {
	  ++nchange;
	}
      }

      next[i*stride + j] = newval;
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

  // Current implementation uses raw pointers for simplicity
  int* state;
  int* tmp;

  // Neighbouring ranks. -1 => no neighbour
  // Ordering is NESW
  std::array<int, 4> neigh_ranks;
  std::array<int, 4> neigh_counts;

  // Halo exchange buffers
  std::vector<int> halo_send_buf;
  std::vector<int> halo_recv_buf;

  Impl(ParallelDecomp const& pd) : p(pd) {
    // Set up device assignment using MPI_Comm_split_type
    MPI_Comm node_comm;
    int node_rank;
    
    MPI_Comm_split_type(p.comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &node_rank);
    
    // Get number of devices - simplified approach
    int num_devices = 1;  // Fallback value
    #pragma omp parallel
    {
      #pragma omp master
      {
        num_devices = omp_get_num_devices();
      }
    }
    
    // Assign device (round-robin)
    device_id = node_rank % num_devices;
    
    // Let's print the assignment for debugging
    printf("Rank %d on node assigned to GPU device %d\n", p.rank, device_id);
    
    // Set default device
    omp_set_default_device(device_id);
    
    // Free the communicator
    MPI_Comm_free(&node_comm);

    // Allocate memory
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
  // halos on the receiving side.
  void halo_exchange(int* data) {
    // Simple halo exchange for single device
    std::array<MPI_Request, 4> reqs;

    auto const stride = p.sub_ny + 2;

    // Post recvs
    for (int b = 0, offset = 0; b < 4; ++b) {
      if (neigh_ranks[b] == -1) {
	reqs[b] = MPI_REQUEST_NULL;
      } else {
	MPI_Irecv(halo_recv_buf.data() + offset, neigh_counts[b], MPI_INT,
		  neigh_ranks[b], 0, p.comm, &reqs[b]);
      }
      offset += neigh_counts[b];
    }

    // Copy halo data from device to host
    #pragma omp target update from(data[0:local_size()])

    // pack the buffers (NESW)
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

    // Send
    for (int b = 0, offset = 0; b < 4; ++b) {
      if (neigh_ranks[b] != -1) {
	MPI_Ssend(halo_send_buf.data() + offset, neigh_counts[b], MPI_INT,
		  neigh_ranks[b], 0, p.comm);
      }
      offset += neigh_counts[b];
    }
    // Wait for receives
    MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

    // Unpack (NESW)
    offset = 0;
    // N
    for (int i = 1; i <= p.sub_nx; ++i, ++offset) {
      data[i * stride + p.sub_ny +1] = halo_recv_buf[offset];
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

    // Copy updated data back to device
    #pragma omp target update to(data[0:local_size()])
  }
};

GpuRunner::GpuRunner(ParallelDecomp const& p) : m_impl(std::make_unique<Impl>(p)) {
}

GpuRunner::~GpuRunner() = default;

// Give each process a sub-array of size (sub_nx+2) x (sub_ny+2)
void GpuRunner::copy_in(int const* source) {
  // Simple host-to-device copy
  std::copy(source, source + m_impl->local_size(), m_impl->state);
  #pragma omp target update to(m_impl->state[0:m_impl->local_size()])
}

// Give each process a sub-array of size (sub_nx+2) x (sub_ny+2)
void GpuRunner::copy_out(int* dest) const {
  // Simple device-to-host copy
  #pragma omp target update from(m_impl->state[0:m_impl->local_size()])
  std::copy(m_impl->state, m_impl->state + m_impl->local_size(), dest);
}

// Iteratively perform percolation of the non-zero elements until no
// changes or 4 *max(M, N) iterations.
void GpuRunner::run() {
  auto& p = m_impl->p;

  int const M = p.nx;
  int const N = p.ny;
  
  // Copy initial state to device
  #pragma omp target update to(m_impl->state[0:m_impl->local_size()])
  std::memcpy(m_impl->tmp, m_impl->state, sizeof(int) * m_impl->local_size());
  #pragma omp target update to(m_impl->tmp[0:m_impl->local_size()])

  int const maxstep = 4 * std::max(M, N);
  int step = 1;
  int global_nchange = 1;

  // Use pointers to the buffers (which we swap below) to avoid copies.
  int* current = m_impl->state;
  int* next = m_impl->tmp;

  while (global_nchange && step <= maxstep) {
    // Ensure edge sites have been communicated to neighbouring
    // processes.
    m_impl->halo_exchange(current);
    
    // Update this process's subdomain
    int local_nchange = percolate_gpu_step(p.sub_nx, p.sub_ny, current, next);

    // Share the total changes to make a decision
    MPI_Allreduce(&local_nchange,
		  &global_nchange,
		  1, MPI_INT, MPI_SUM, p.comm);

    if (step % printfreq == 0) {
      p.print_root("percolate: number of changes on step {} is {}\n",
		   step, global_nchange);
    }

    // Swap the pointers for the next iteration
    std::swap(next, current);
    step++;
  }

  // Ensure final state is on host
  #pragma omp target update from(current[0:m_impl->local_size()])
  
  // Answer now in `current`, make sure this one is in `state`
  if (current != m_impl->state) {
    std::memcpy(m_impl->state, current, sizeof(int) * m_impl->local_size());
  }
}