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

#include <memory>
#include <algorithm>
#include <numeric>
#include <vector>
#include <array>
#include <cstring>
#include <unistd.h> // For gethostname
#include "util.h"
#include "decomp.h"
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

  // OpenMP target offload to GPU
  #pragma omp target teams distribute parallel for collapse(2) map(to: state[0:(M+2)*(N+2)]) map(from: next[0:(M+2)*(N+2)]) map(tofrom: nchange) reduction(+:nchange)
  for (int i = 1; i <= M; ++i) {
    for (int j = 1; j <= N; ++j) {
      int const idx = i*stride + j;
      int const oldval = state[idx];
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

      next[idx] = newval;
    }
  }
  return nchange;
}


struct GpuRunner::Impl {
  // Here you can store any parameters or data needed for
  // implementation of your version.

  ParallelDecomp p;

  // Would perhaps be better to use vector or at least a unique_ptr,
  // but want to borrow pointers, swap, then set them back.
  int* state;
  int* tmp;

  // GPU related variables
  MPI_Comm node_comm;  // Communicator for processes on the same node
  int node_rank;       // Rank within the node
  int device_id;       // Assigned GPU device ID
  bool using_gpu;      // Flag indicating if GPU is being used

  // Neighbouring ranks. -1 => no neighbour
  // Ordering is NESW
  std::array<int, 4> neigh_ranks;
  std::array<int, 4> neigh_counts;

  // For a simple halo exchange are going to double buffer.  Copy the
  // potentially strided data out of the array, send/recv it and then
  // copy it out on the other side.
  //
  // Allocate enough space for all 4 directions in c'tor, even if all
  // directions not needed.
  std::vector<int> halo_send_buf;
  std::vector<int> halo_recv_buf;

  Impl(ParallelDecomp const& pd) : p(pd) {
    // First, create a communicator for processes on the same node
    MPI_Comm_split_type(p.comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &node_rank);
    
    // Get number of MPI processes on this node
    int node_size;
    MPI_Comm_size(node_comm, &node_size);
    
    // Query available GPU devices
    int num_devices = omp_get_num_devices();
    
    // Set default to not using GPU
    using_gpu = false;
    device_id = -1;
    
    // Print basic information from all processes
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    
    // Gather information from all processes to print on root
    struct {
      int global_rank;
      int node_rank;
      int num_devices;
      int assigned_device;
      char hostname[256];
    } my_info, *all_info = nullptr;
    
    // Fill in local info
    my_info.global_rank = p.rank;
    my_info.node_rank = node_rank;
    my_info.num_devices = num_devices;
    my_info.assigned_device = -1;  // Will be set later
    strncpy(my_info.hostname, hostname, sizeof(my_info.hostname));
    
    // Allocate buffer on root process
    int world_size;
    MPI_Comm_size(p.comm, &world_size);
    if (p.rank == 0) {
        all_info = new decltype(my_info)[world_size];
    }
    
    if (num_devices > 0) {
      // We have GPUs available
      // Assign GPU in round-robin fashion within the node (4 GPUs per node)
      device_id = node_rank % num_devices;
      my_info.assigned_device = device_id;
      
      // Set this device as the default for OpenMP
      omp_set_default_device(device_id);
      
      // Test if the GPU is actually usable
      int is_device_usable = 0;
      #pragma omp target map(from:is_device_usable)
      {
        is_device_usable = 1;  // This will only execute if the device is accessible
      }
      
      if (is_device_usable) {
        using_gpu = true;
      } else {
        device_id = -1;
        my_info.assigned_device = -1;
        using_gpu = false;
      }
    }
    
    // Gather all information to root process
    MPI_Gather(&my_info, sizeof(my_info), MPI_BYTE, 
               all_info, sizeof(my_info), MPI_BYTE, 
               0, p.comm);
    
    // Print summary on root process
    if (p.rank == 0) {
        printf("\n==== GPU Assignment Summary ====\n");
        printf("%-6s %-10s %-10s %-8s %-8s %s\n", 
               "Rank", "Node Rank", "Hostname", "GPUs", "Using", "Status");
        printf("-----------------------------------------------------------\n");
        
        for (int i = 0; i < world_size; i++) {
            printf("%-6d %-10d %-10s %-8d %-8d %s\n", 
                   all_info[i].global_rank, 
                   all_info[i].node_rank,
                   all_info[i].hostname,
                   all_info[i].num_devices,
                   all_info[i].assigned_device,
                   all_info[i].assigned_device >= 0 ? "GPU" : "CPU (fallback)");
        }
        printf("==== End GPU Assignment Summary ====\n\n");
        
        // Clean up
        delete[] all_info;
    }

    state = new int[local_size()];
    tmp = new int[local_size()];

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

    // Allocate for all of them. Need two for x and 2 for y
    auto halo_buf_size = 2 * (p.sub_nx + p.sub_ny);
    halo_send_buf.resize(halo_buf_size);
    halo_recv_buf.resize(halo_buf_size);
    
    // If we're using GPU, allocate device memory
    if (using_gpu) {
      #pragma omp target enter data map(alloc: state[0:local_size()], tmp[0:local_size()])
    }
  }
    
  ~Impl() {
    // Free GPU memory if we were using it
    if (using_gpu) {
      #pragma omp target exit data map(delete: state[0:local_size()], tmp[0:local_size()])
    }
    
    delete[] state;
    delete[] tmp;
    
    // Free the node communicator
    MPI_Comm_free(&node_comm);
  }

  int local_size() const {
    return (p.sub_nx + 2)*(p.sub_ny + 2);
  }

  // Communicate the valid edge sites to neighbours that exist. Fill
  // halos on the receiving side.
  void halo_exchange(int* data) {
    // For GPU execution, first copy from device to host
    if (using_gpu) {
      #pragma omp target update from(data[0:local_size()])
    }
    
    // Recall: order of directions is NESW.
    // 
    // Strategy: post non-blocking receives into recv buf, pack send
    // buf, send send buf, wait, unpack recv buf.
    std::array<MPI_Request, 8> reqs;  // 4 for recv, 4 for send

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

    // Send with non-blocking operations
    for (int b = 0, offset = 0; b < 4; ++b) {
      if (neigh_ranks[b] != -1) {
        MPI_Isend(halo_send_buf.data() + offset, neigh_counts[b], MPI_INT,
                 neigh_ranks[b], 0, p.comm, &reqs[b+4]);
      } else {
        reqs[b+4] = MPI_REQUEST_NULL;
      }
      offset += neigh_counts[b];
    }
    
    // Wait for receives to complete
    for (int b = 0; b < 4; ++b) {
      if (neigh_ranks[b] != -1) {
        MPI_Wait(&reqs[b], MPI_STATUS_IGNORE);
      }
    }

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
    
    // Wait for sends to complete
    for (int b = 4; b < 8; ++b) {
      if (neigh_ranks[b-4] != -1) {
        MPI_Wait(&reqs[b], MPI_STATUS_IGNORE);
      }
    }
    
    // For GPU execution, update device with new halo data
    if (using_gpu) {
      #pragma omp target update to(data[0:local_size()])
    }
  }
};

GpuRunner::GpuRunner(ParallelDecomp const& p) : m_impl(std::make_unique<Impl>(p)) {
}

GpuRunner::~GpuRunner() = default;

// Give each process a sub-array of size (sub_nx+2) x (sub_ny+2)
void GpuRunner::copy_in(int const* source) {
  std::copy(source, source + m_impl->local_size(), m_impl->state);
  
  // For GPU execution, copy data to device
  if (m_impl->using_gpu) {
    #pragma omp target update to(m_impl->state[0:m_impl->local_size()])
  }
}

// Give each process a sub-array of size (sub_nx+2) x (sub_ny+2)
void GpuRunner::copy_out(int* dest) const {
  // For GPU execution, copy data from device
  if (m_impl->using_gpu) {
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
  
  // Log GPU utilization at the start
  if (p.rank == 0) {
    // Count total GPUs being used
    int total_used_gpus = 0;
    int using_gpu_val = m_impl->using_gpu ? 1 : 0;
    MPI_Reduce(&using_gpu_val, &total_used_gpus, 1, MPI_INT, MPI_SUM, 0, p.comm);
    
    // Get world size
    int world_size;
    MPI_Comm_size(p.comm, &world_size);
    
    printf("\n==== GPU Utilization ====\n");
    printf("Total processes: %d\n", world_size);
    printf("Processes using GPU: %d (%.1f%%)\n", total_used_gpus, 
           (100.0 * total_used_gpus) / world_size);
    printf("========================\n\n");
  } else {
    // Non-root processes just participate in the reduction
    int using_gpu = m_impl->using_gpu ? 1 : 0;
    MPI_Reduce(&using_gpu, nullptr, 1, MPI_INT, MPI_SUM, 0, p.comm);
  }
  
  // Copy the initial state to the tmp, only the global halos are
  // *required*, but much easier this way!
  std::memcpy(m_impl->tmp, m_impl->state, sizeof(int) * m_impl->local_size());
  
  // For GPU execution, update device memory
  if (m_impl->using_gpu) {
    #pragma omp target update to(m_impl->tmp[0:m_impl->local_size()])
  }

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
    
    // Update this process's subdomain - GPU or CPU implementation
    int local_nchange;
    
    if (m_impl->using_gpu) {
      // Use GPU implementation
      local_nchange = percolate_gpu_step(p.sub_nx, p.sub_ny, current, next);
    } else {
      // Use CPU implementation
      local_nchange = 0;
      int const stride = p.sub_ny + 2;
      
      for (int i = 1; i <= p.sub_nx; ++i) {
        for (int j = 1; j <= p.sub_ny; ++j) {
          int const idx = i*stride + j;
          int const oldval = current[idx];
          int newval = oldval;

          // 0 => solid, so do nothing
          if (oldval != 0) {
            // Set next[i][j] to be the maximum value of state[i][j] and
            // its four nearest neighbours
            newval = std::max(newval, current[(i-1)*stride + j  ]);
            newval = std::max(newval, current[(i+1)*stride + j  ]);
            newval = std::max(newval, current[    i*stride + j-1]);
            newval = std::max(newval, current[    i*stride + j+1]);

            if (newval != oldval) {
              ++local_nchange;
            }
          }

          next[idx] = newval;
        }
      }
    }

    // Share the total changes to make a decision
    MPI_Allreduce(&local_nchange,
         &global_nchange,
         1, MPI_INT, MPI_SUM, p.comm);

    //  Report progress every now and then
    if (step % printfreq == 0) {
      p.print_root("percolate: number of changes on step {} is {}\n",
         step, global_nchange);
    }

    // Swap the pointers for the next iteration
    std::swap(next, current);
    step++;
  }

  // Answer now in `current`, make sure this one is in `state`
  if (current != m_impl->state) {
    // Copy data from current to state
    std::memcpy(m_impl->state, current, sizeof(int) * m_impl->local_size());
    
    // For GPU execution, ensure state is updated on device
    if (m_impl->using_gpu) {
      #pragma omp target update to(m_impl->state[0:m_impl->local_size()])
    }
  }
  
  // Update pointers
  m_impl->state = current;
  m_impl->tmp = next;
  
  // Log final step count by GPU/CPU usage
  if (p.rank == 0) {
    printf("\n==== Percolation Complete ====\n");
    printf("Completed in %d steps (max allowed: %d)\n", step, maxstep);
    printf("============================\n\n");
  }
}
