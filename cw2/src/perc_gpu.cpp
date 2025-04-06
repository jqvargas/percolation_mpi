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

  // Enhanced GPU kernel with explicit mapping
  #pragma omp target teams distribute parallel for collapse(2) \
    reduction(+:nchange)
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
  ParallelDecomp p;
  int* state;
  int* tmp;

  // Add node communicator and device management
  MPI_Comm node_comm;
  int node_rank;
  int device_id;

  std::array<int, 4> neigh_ranks;
  std::array<int, 4> neigh_counts;

  // Separate buffers for each boundary
  std::array<std::vector<int>, 4> halo_send_bufs;
  std::array<std::vector<int>, 4> halo_recv_bufs;

  // MPI request handles for non-blocking communication
  std::array<MPI_Request, 8> requests;  // 4 for receives, 4 for sends

  Impl(ParallelDecomp const& pd) : p(pd) {
    // Create a communicator for processes on the same node
    MPI_Comm_split_type(p.comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &node_rank);

    // Get number of available GPUs
    int num_devices = omp_get_num_devices();
    
    // Be more defensive about GPU availability
    if (num_devices == 0) {
      p.print_root("Warning: No GPU devices available, falling back to CPU\n");
      device_id = -1;
    } else {
      // Assign GPU in round-robin fashion within each node
      device_id = node_rank % num_devices;
      
      // Check if we can actually use this device
      omp_set_default_device(device_id);
      if (!omp_is_initial_device()) {
        p.print_root("Process {} on node {} assigned to GPU {}\n", p.rank, node_rank, device_id);
      } else {
        p.print_root("Process {} on node {} failed to get GPU {}, falling back to CPU\n", p.rank, node_rank, device_id);
        device_id = -1;
      }
    }

    // Allocate host memory
    state = new int[local_size()];
    tmp = new int[local_size()];
    
    // Initialize host memory to zero
    std::memset(state, 0, local_size() * sizeof(int));
    std::memset(tmp, 0, local_size() * sizeof(int));

    // Initialize device memory if GPU is available
    if (device_id >= 0) {
      #pragma omp target enter data map(alloc: state[0:local_size()])
      #pragma omp target enter data map(alloc: tmp[0:local_size()])
      
      // Explicitly copy initialized data to device
      #pragma omp target update to(state[0:local_size()], tmp[0:local_size()])
    }

    // Neighbor ranks and counts
    neigh_ranks[0] = (p.pj + 1 < p.py) ? p.global_rank(p.pi, p.pj + 1) : -1; // top
    neigh_counts[0] = p.sub_nx;
    neigh_ranks[1] = (p.pi + 1 < p.px) ? p.global_rank(p.pi + 1, p.pj) : -1; // right
    neigh_counts[1] = p.sub_ny;
    neigh_ranks[2] = (p.pj - 1 >= 0) ? p.global_rank(p.pi, p.pj - 1) : -1; // bottom
    neigh_counts[2] = p.sub_nx;
    neigh_ranks[3] = (p.pi - 1 >= 0) ? p.global_rank(p.pi - 1, p.pj) : -1; // left
    neigh_counts[3] = p.sub_ny;

    // Initialize separate buffers for each boundary
    for (int i = 0; i < 4; ++i) {
      if (neigh_ranks[i] != -1) {
        halo_send_bufs[i].resize(neigh_counts[i]);
        halo_recv_bufs[i].resize(neigh_counts[i]);
      }
    }

    // Initialize MPI requests
    for (auto& req : requests) {
      req = MPI_REQUEST_NULL;
    }
  }
    
  ~Impl() {
    // Clean up device memory if GPU was used
    if (device_id >= 0) {
      try {
        #pragma omp target exit data map(delete: state[0:local_size()])
        #pragma omp target exit data map(delete: tmp[0:local_size()])
      } catch (...) {
        // Ignore errors during cleanup
        p.print_root("Warning: Error during GPU memory cleanup\n");
      }
    }
    delete[] state;
    delete[] tmp;
    MPI_Comm_free(&node_comm);
  }

  int local_size() const {
    return (p.sub_nx + 2)*(p.sub_ny + 2);
  }

  // Communicate the valid edge sites to neighbours that exist. Fill
  // halos on the receiving side using non-blocking communication
  void halo_exchange(int* data) {
    auto const stride = p.sub_ny + 2;
    
    // Handle data differently depending on whether we're using GPU or CPU
    if (device_id >= 0) {
      // GPU path: copy from device to host
      std::vector<int> host_data(local_size());
      #pragma omp target update from(data[0:local_size()])
      std::copy(data, data + local_size(), host_data.data());
      
      // Post receives first
      for (int b = 0; b < 4; ++b) {
        if (neigh_ranks[b] != -1) {
          MPI_Irecv(halo_recv_bufs[b].data(), neigh_counts[b], MPI_INT,
                  neigh_ranks[b], 0, p.comm, &requests[b]);
        } else {
          requests[b] = MPI_REQUEST_NULL;
        }
      }
      
      // Pack boundary data from host_data into send buffers
      if (neigh_ranks[0] != -1) { // top
        for (int i = 0; i < p.sub_nx; ++i) {
          halo_send_bufs[0][i] = host_data[(i+1)*stride + p.sub_ny];
        }
      }
      if (neigh_ranks[1] != -1) { // right
        for (int j = 0; j < p.sub_ny; ++j) {
          halo_send_bufs[1][j] = host_data[p.sub_nx*stride + (j+1)];
        }
      }
      if (neigh_ranks[2] != -1) { // bottom
        for (int i = 0; i < p.sub_nx; ++i) {
          halo_send_bufs[2][i] = host_data[(i+1)*stride + 1];
        }
      }
      if (neigh_ranks[3] != -1) { // left
        for (int j = 0; j < p.sub_ny; ++j) {
          halo_send_bufs[3][j] = host_data[1*stride + (j+1)];
        }
      }
      
      // Send boundary data
      for (int b = 0; b < 4; ++b) {
        if (neigh_ranks[b] != -1) {
          MPI_Isend(halo_send_bufs[b].data(), neigh_counts[b], MPI_INT,
                  neigh_ranks[b], 0, p.comm, &requests[b+4]);
        } else {
          requests[b+4] = MPI_REQUEST_NULL;
        }
      }
      
      // Wait for receives to complete
      for (int b = 0; b < 4; ++b) {
        if (neigh_ranks[b] != -1) {
          MPI_Wait(&requests[b], MPI_STATUS_IGNORE);
        }
      }
      
      // Unpack received data into host_data
      if (neigh_ranks[0] != -1) { // top
        for (int i = 0; i < p.sub_nx; ++i) {
          host_data[(i+1)*stride + p.sub_ny + 1] = halo_recv_bufs[0][i];
        }
      }
      if (neigh_ranks[1] != -1) { // right
        for (int j = 0; j < p.sub_ny; ++j) {
          host_data[(p.sub_nx + 1)*stride + (j+1)] = halo_recv_bufs[1][j];
        }
      }
      if (neigh_ranks[2] != -1) { // bottom
        for (int i = 0; i < p.sub_nx; ++i) {
          host_data[(i+1)*stride + 0] = halo_recv_bufs[2][i];
        }
      }
      if (neigh_ranks[3] != -1) { // left
        for (int j = 0; j < p.sub_ny; ++j) {
          host_data[0*stride + (j+1)] = halo_recv_bufs[3][j];
        }
      }
      
      // Copy the updated data back to the original array and to the device
      std::copy(host_data.data(), host_data.data() + local_size(), data);
      #pragma omp target update to(data[0:local_size()])
      
      // Wait for sends to complete
      for (int b = 4; b < 8; ++b) {
        if (neigh_ranks[b-4] != -1) {
          MPI_Wait(&requests[b], MPI_STATUS_IGNORE);
        }
      }
    } 
    else {
      // CPU path: work directly with the data
      
      // Post receives first
      for (int b = 0; b < 4; ++b) {
        if (neigh_ranks[b] != -1) {
          MPI_Irecv(halo_recv_bufs[b].data(), neigh_counts[b], MPI_INT,
                  neigh_ranks[b], 0, p.comm, &requests[b]);
        } else {
          requests[b] = MPI_REQUEST_NULL;
        }
      }
      
      // Pack boundary data directly from data array
      if (neigh_ranks[0] != -1) { // top
        for (int i = 0; i < p.sub_nx; ++i) {
          halo_send_bufs[0][i] = data[(i+1)*stride + p.sub_ny];
        }
      }
      if (neigh_ranks[1] != -1) { // right
        for (int j = 0; j < p.sub_ny; ++j) {
          halo_send_bufs[1][j] = data[p.sub_nx*stride + (j+1)];
        }
      }
      if (neigh_ranks[2] != -1) { // bottom
        for (int i = 0; i < p.sub_nx; ++i) {
          halo_send_bufs[2][i] = data[(i+1)*stride + 1];
        }
      }
      if (neigh_ranks[3] != -1) { // left
        for (int j = 0; j < p.sub_ny; ++j) {
          halo_send_bufs[3][j] = data[1*stride + (j+1)];
        }
      }
      
      // Send boundary data
      for (int b = 0; b < 4; ++b) {
        if (neigh_ranks[b] != -1) {
          MPI_Isend(halo_send_bufs[b].data(), neigh_counts[b], MPI_INT,
                  neigh_ranks[b], 0, p.comm, &requests[b+4]);
        } else {
          requests[b+4] = MPI_REQUEST_NULL;
        }
      }
      
      // Wait for receives to complete
      for (int b = 0; b < 4; ++b) {
        if (neigh_ranks[b] != -1) {
          MPI_Wait(&requests[b], MPI_STATUS_IGNORE);
        }
      }
      
      // Unpack received data directly into data array
      if (neigh_ranks[0] != -1) { // top
        for (int i = 0; i < p.sub_nx; ++i) {
          data[(i+1)*stride + p.sub_ny + 1] = halo_recv_bufs[0][i];
        }
      }
      if (neigh_ranks[1] != -1) { // right
        for (int j = 0; j < p.sub_ny; ++j) {
          data[(p.sub_nx + 1)*stride + (j+1)] = halo_recv_bufs[1][j];
        }
      }
      if (neigh_ranks[2] != -1) { // bottom
        for (int i = 0; i < p.sub_nx; ++i) {
          data[(i+1)*stride + 0] = halo_recv_bufs[2][i];
        }
      }
      if (neigh_ranks[3] != -1) { // left
        for (int j = 0; j < p.sub_ny; ++j) {
          data[0*stride + (j+1)] = halo_recv_bufs[3][j];
        }
      }
      
      // Wait for sends to complete
      for (int b = 4; b < 8; ++b) {
        if (neigh_ranks[b-4] != -1) {
          MPI_Wait(&requests[b], MPI_STATUS_IGNORE);
        }
      }
    }
  }
};

GpuRunner::GpuRunner(ParallelDecomp const& p) : m_impl(std::make_unique<Impl>(p)) {
}

GpuRunner::~GpuRunner() = default;

// Give each process a sub-array of size (sub_nx+2) x (sub_ny+2)
void GpuRunner::copy_in(int const* source) {
  // Copy data from host to our buffer
  std::copy(source, source + m_impl->local_size(), m_impl->state);
  
  // Only transfer to device if we're using a GPU
  if (m_impl->device_id >= 0) {
    #pragma omp target update to(m_impl->state[0:m_impl->local_size()])
  }
}

// Give each process a sub-array of size (sub_nx+2) x (sub_ny+2)
void GpuRunner::copy_out(int* dest) const {
  // Only transfer from device if we're using a GPU
  if (m_impl->device_id >= 0) {
    #pragma omp target update from(m_impl->state[0:m_impl->local_size()])
  }
  std::copy(m_impl->state, m_impl->state + m_impl->local_size(), dest);
}

// Iteratively perform percolation of the non-zero elements until no
// changes or 4 *max(M, N) iterations.
void GpuRunner::run() {
  auto& p = m_impl->p;
  int const M = p.nx;
  int const N = p.ny;
  
  // If no GPU is available, run on CPU
  if (m_impl->device_id < 0) {
    p.print_root("Process {} running on CPU\n", p.rank);
    
    int const maxstep = 4 * std::max(M, N);
    int step = 1;
    int global_nchange = 1;
    
    // Use direct pointers for CPU execution
    int* current = m_impl->state;
    int* next = m_impl->tmp;
    
    while (global_nchange && step <= maxstep) {
      // Exchange halos with neighboring processes (CPU version)
      m_impl->halo_exchange(current);
      
      // CPU implementation of percolate step
      int local_nchange = 0;
      int const stride = p.sub_ny + 2;
      
      // Sequential computation
      for (int i = 1; i <= p.sub_nx; ++i) {
        for (int j = 1; j <= p.sub_ny; ++j) {
          int const idx = i*stride + j;
          int const oldval = current[idx];
          int newval = oldval;
          
          if (oldval != 0) {
            newval = std::max(newval, current[idx - stride]); // North
            newval = std::max(newval, current[idx + stride]); // South
            newval = std::max(newval, current[idx - 1]);      // West
            newval = std::max(newval, current[idx + 1]);      // East
            
            if (newval != oldval) {
              ++local_nchange;
            }
          }
          
          next[idx] = newval;
        }
      }
      
      // Share the total changes
      MPI_Allreduce(&local_nchange, &global_nchange, 1, MPI_INT, MPI_SUM, p.comm);
      
      if (step % printfreq == 0 && p.rank == 0) {
        p.print_root("percolate: number of changes on step {} is {}\n", step, global_nchange);
      }
      
      // Swap pointers for next iteration
      std::swap(current, next);
      step++;
    }
    
    // Ensure final state is in m_impl->state
    if (current != m_impl->state) {
      std::copy(current, current + m_impl->local_size(), m_impl->state);
    }
  } 
  else { // GPU execution path
    p.print_root("Process {} running on GPU {}\n", p.rank, m_impl->device_id);
    
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
  
      // Instead of swapping pointers, always use fixed arrays and copy data
      bool use_state_as_current = true;
  
      try {
        while (global_nchange && step <= maxstep) {
          // Determine which arrays to use as current and next
          int* current = use_state_as_current ? m_impl->state : m_impl->tmp;
          int* next = use_state_as_current ? m_impl->tmp : m_impl->state;
  
          // Exchange halos with neighboring processes
          m_impl->halo_exchange(current);
          
          // Run the computation kernel
          int local_nchange = percolate_gpu_step(p.sub_nx, p.sub_ny, current, next);
  
          // Share the total changes
          MPI_Allreduce(&local_nchange, &global_nchange, 1, MPI_INT, MPI_SUM, p.comm);
  
          if (step % printfreq == 0 && p.rank == 0) {
            p.print_root("percolate: number of changes on step {} is {}\n", step, global_nchange);
          }
  
          // Toggle which array to use as current for next iteration
          use_state_as_current = !use_state_as_current;
          step++;
        }
  
        // Ensure final state is in m_impl->state
        if (!use_state_as_current) {
          #pragma omp target teams distribute parallel for collapse(2)
          for (int i = 0; i < p.sub_nx + 2; ++i) {
            for (int j = 0; j < p.sub_ny + 2; ++j) {
              m_impl->state[i*(p.sub_ny+2) + j] = m_impl->tmp[i*(p.sub_ny+2) + j];
            }
          }
        }
      } catch (const std::exception& e) {
        p.print_root("Error during GPU execution: {}\n", e.what());
        throw; // Re-throw to ensure clean exit
      }
    } // End of target data region
  }
}