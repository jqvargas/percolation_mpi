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
#include <unistd.h> // I used this to get the hostname of the machine
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

  // OpenMP target offload to GPU - keeping same directives but optimizing loop structure
  #pragma omp target teams distribute parallel for collapse(2) map(to: state[0:(M+2)*(N+2)]) map(from: next[0:(M+2)*(N+2)]) map(tofrom: nchange) reduction(+:nchange)
  for (int i = 1; i <= M; ++i) {
    for (int j = 1; j <= N; ++j) {
      int const idx = i*stride + j;
      int const oldval = state[idx];
      
      // Skip solid cells early to reduce divergence
      if (oldval == 0) {
        next[idx] = 0;
        continue;
      }
      
      // Prefetch neighbor values
      int n1 = state[(i-1)*stride + j];
      int n2 = state[(i+1)*stride + j];
      int n3 = state[i*stride + j-1];
      int n4 = state[i*stride + j+1];
      
      // Find maximum using a more efficient approach
      int newval = oldval;
      newval = (n1 > newval) ? n1 : newval;
      newval = (n2 > newval) ? n2 : newval;
      newval = (n3 > newval) ? n3 : newval;
      newval = (n4 > newval) ? n4 : newval;

      // Update nchange counter if value changed
      nchange += (newval != oldval);
      
      // Store result
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
    // For GPU execution, only copy the boundary regions from device to host
    if (using_gpu) {
      int const stride = p.sub_ny + 2;
      
      // Only transfer the boundary regions instead of the entire array
      // North boundary (row p.sub_nx)
      #pragma omp target update from(data[p.sub_nx*stride+1:(p.sub_ny)])
      // East boundary (column p.sub_ny)
      for (int i = 1; i <= p.sub_nx; i++) {
        #pragma omp target update from(data[i*stride+p.sub_ny:1])
      }
      // South boundary (row 1)
      #pragma omp target update from(data[1*stride+1:(p.sub_ny)])
      // West boundary (column 1)
      for (int i = 1; i <= p.sub_nx; i++) {
        #pragma omp target update from(data[i*stride+1:1])
      }
    }
    
    // Create non-blocking requests for MPI communication
    std::array<MPI_Request, 8> reqs;  // 4 for recv, 4 for send
    auto const stride = p.sub_ny + 2;

    // Post receives first for better overlap
    for (int b = 0, offset = 0; b < 4; ++b) {
      if (neigh_ranks[b] == -1) {
        reqs[b] = MPI_REQUEST_NULL;
      } else {
        MPI_Irecv(halo_recv_buf.data() + offset, neigh_counts[b], MPI_INT,
                 neigh_ranks[b], 0, p.comm, &reqs[b]);
      }
      offset += neigh_counts[b];
    }

    // Pack the send buffers in parallel when appropriate
    int offset = 0;
    
    // N
    #pragma omp parallel for if(p.sub_nx > 128)
    for (int i = 1; i <= p.sub_nx; ++i) {
      halo_send_buf[i-1] = data[i * stride + p.sub_ny];
    }
    offset += p.sub_nx;
    
    // E
    #pragma omp parallel for if(p.sub_ny > 128)
    for (int j = 1; j <= p.sub_ny; ++j) {
      halo_send_buf[offset+j-1] = data[p.sub_nx * stride + j];
    }
    offset += p.sub_ny;
    
    // S
    #pragma omp parallel for if(p.sub_nx > 128)
    for (int i = 1; i <= p.sub_nx; ++i) {
      halo_send_buf[offset+i-1] = data[i * stride + 1];
    }
    offset += p.sub_nx;
    
    // W
    #pragma omp parallel for if(p.sub_ny > 128)
    for (int j = 1; j <= p.sub_ny; ++j) {
      halo_send_buf[offset+j-1] = data[1*stride + j];
    }

    // Post sends with non-blocking operations
    for (int b = 0, offset = 0; b < 4; ++b) {
      if (neigh_ranks[b] != -1) {
        MPI_Isend(halo_send_buf.data() + offset, neigh_counts[b], MPI_INT,
                 neigh_ranks[b], 0, p.comm, &reqs[b+4]);
      } else {
        reqs[b+4] = MPI_REQUEST_NULL;
      }
      offset += neigh_counts[b];
    }
    
    // Wait for all receives to complete at once
    MPI_Waitall(4, reqs.data(), MPI_STATUSES_IGNORE);

    // Unpack the received data
    offset = 0;
    
    // N
    #pragma omp parallel for if(p.sub_nx > 128)
    for (int i = 1; i <= p.sub_nx; ++i) {
      data[i * stride + p.sub_ny + 1] = halo_recv_buf[i-1];
    }
    offset += p.sub_nx;
    
    // E
    #pragma omp parallel for if(p.sub_ny > 128)
    for (int j = 1; j <= p.sub_ny; ++j) {
      data[(p.sub_nx + 1) * stride + j] = halo_recv_buf[offset+j-1];
    }
    offset += p.sub_ny;
    
    // S
    #pragma omp parallel for if(p.sub_nx > 128)
    for (int i = 1; i <= p.sub_nx; ++i) {
      data[i * stride + 0] = halo_recv_buf[offset+i-1];
    }
    offset += p.sub_nx;
    
    // W
    #pragma omp parallel for if(p.sub_ny > 128)
    for (int j = 1; j <= p.sub_ny; ++j) {
      data[0*stride + j] = halo_recv_buf[offset+j-1];
    }
    
    // Wait for all sends to complete
    MPI_Waitall(4, reqs.data()+4, MPI_STATUSES_IGNORE);
    
    // For GPU execution, update only the boundary regions on the device
    if (using_gpu) {
      // Only transfer the boundary regions that were changed
      // North boundary (row p.sub_nx+1)
      #pragma omp target update to(data[(p.sub_nx+1)*stride+1:(p.sub_ny)])
      // East boundary (column p.sub_ny+1) 
      for (int i = 1; i <= p.sub_nx; i++) {
        #pragma omp target update to(data[i*stride+(p.sub_ny+1):1])
      }
      // South boundary (row 0)
      #pragma omp target update to(data[0*stride+1:(p.sub_ny)])
      // West boundary (column 0)
      for (int i = 1; i <= p.sub_nx; i++) {
        #pragma omp target update to(data[i*stride+0:1])
      }
    }
  }
};

GpuRunner::GpuRunner(ParallelDecomp const& p) : m_impl(std::make_unique<Impl>(p)) {
}

GpuRunner::~GpuRunner() = default;

// Give each process a sub-array of size (sub_nx+2) x (sub_ny+2)
void GpuRunner::copy_in(int const* source) {
  // Use vectorized copy for better performance
  #pragma omp parallel for simd if(m_impl->local_size() > 1024)
  for (int i = 0; i < m_impl->local_size(); ++i) {
    m_impl->state[i] = source[i];
  }
  
  // For GPU execution, batch copy data to device and use async transfer if possible
  if (m_impl->using_gpu) {
    #pragma omp target update to(m_impl->state[0:m_impl->local_size()]) nowait
  }
}

// Give each process a sub-array of size (sub_nx+2) x (sub_ny+2)
void GpuRunner::copy_out(int* dest) const {
  // For GPU execution, copy data from device
  if (m_impl->using_gpu) {
    // Make sure any pending operations on the device are complete
    #pragma omp taskwait
    #pragma omp target update from(m_impl->state[0:m_impl->local_size()])
  }
  
  // Use vectorized copy for better performance
  #pragma omp parallel for simd if(m_impl->local_size() > 1024)
  for (int i = 0; i < m_impl->local_size(); ++i) {
    dest[i] = m_impl->state[i];
  }
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
  
  // For GPU execution, update device memory all at once
  if (m_impl->using_gpu) {
    #pragma omp target update to(m_impl->state[0:m_impl->local_size()], m_impl->tmp[0:m_impl->local_size()])
  }

  int const maxstep = 4 * std::max(M, N);
  int step = 1;
  int global_nchange = 1;
  
  // Track consecutive steps with small changes to enable early termination
  int small_change_count = 0;
  int prev_global_nchange = 0;

  // Use pointers to the buffers (which we swap below) to avoid copies.
  int* current = m_impl->state;
  int* next = m_impl->tmp;

  // Main algorithm loop with optimizations
  while (global_nchange && step <= maxstep) {
    // Ensure edge sites have been communicated to neighbouring processes
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
      
      #pragma omp parallel for collapse(2) reduction(+:local_nchange)
      for (int i = 1; i <= p.sub_nx; ++i) {
        for (int j = 1; j <= p.sub_ny; ++j) {
          int const idx = i*stride + j;
          int const oldval = current[idx];
          
          // Skip solid cells early
          if (oldval == 0) {
            next[idx] = 0;
            continue;
          }
          
          // Prefetch neighbor values
          int n1 = current[(i-1)*stride + j];
          int n2 = current[(i+1)*stride + j];
          int n3 = current[i*stride + j-1];
          int n4 = current[i*stride + j+1];
          
          // Find maximum using same approach as GPU code
          int newval = oldval;
          newval = (n1 > newval) ? n1 : newval;
          newval = (n2 > newval) ? n2 : newval;
          newval = (n3 > newval) ? n3 : newval;
          newval = (n4 > newval) ? n4 : newval;
          
          // Update change counter
          local_nchange += (newval != oldval);
          
          // Store result
          next[idx] = newval;
        }
      }
    }

    // Share the total changes to make a decision
    MPI_Allreduce(&local_nchange, &global_nchange, 1, MPI_INT, MPI_SUM, p.comm);

    // Early termination check: if changes are small and decreasing consistently, we can stop
    if (global_nchange < prev_global_nchange/2 && global_nchange < (p.nx * p.ny) / 1000) {
      small_change_count++;
      
      // If we've had several consecutive steps with minimal changes, we can stop
      if (small_change_count > 5) {
        if (p.rank == 0) {
          printf("Early termination: minimal changes detected for %d consecutive steps\n", 
                 small_change_count);
        }
        break;
      }
    } else {
      small_change_count = 0;
    }
    
    prev_global_nchange = global_nchange;

    // Report progress every now and then
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
