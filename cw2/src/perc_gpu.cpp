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

// Required libraries for smart pointers, container classes, and algorithm operations
#include <memory>
#include <algorithm>
#include <numeric>
#include <vector>
#include <array>
#include <cstring>
#include <unistd.h> // I used this to get the hostname of the machine
#include "util.h"    // Utility functions for the percolation simulation
#include "decomp.h"  // Domain decomposition handling for parallel execution
#include "perc_gpu.h"

#include <cstdio>
#include <cstring>
#include <vector>
#include <omp.h>     // OpenMP support for GPU offloading

#include "decomp.h"

// Frequency for reporting progress during simulation
constexpr int printfreq = 100;

// Perform a single step of the algorithm.
//
// For each point (if fluid), set it to the maximum of itself and the
// four von Neumann neighbours.
//
// Returns the total number of changed cells.
static int percolate_gpu_step(int M, int N, int const* state, int* next) {
  int nchange = 0;
  int const stride = N + 2;  // Add 2 for the ghost cells (halo regions)

  // OpenMP target offload to GPU - simplified for compatibility
  // This directive:
  // 1. Maps necessary data to the GPU
  // 2. Distributes work across multiple teams of threads
  // 3. Parallelizes both dimensions with collapse(2)
  // 4. Accumulates changes via reduction
  #pragma omp target teams distribute parallel for collapse(2) map(to: state[0:(M+2)*(N+2)]) map(from: next[0:(M+2)*(N+2)]) map(tofrom: nchange) reduction(+:nchange)
  for (int i = 1; i <= M; ++i) {
    for (int j = 1; j <= N; ++j) {
      int const idx = i*stride + j;        // Linearized 2D index
      int const oldval = state[idx];
      int newval = oldval;

      // 0 => solid, so do nothing (optimization to skip solid cells)
      if (oldval != 0) {
        // Set next[i][j] to be the maximum value of state[i][j] and
        // its four nearest neighbours (North, South, East, West)
        newval = std::max(newval, state[(i-1)*stride + j]);    // North neighbor
        newval = std::max(newval, state[(i+1)*stride + j]);    // South neighbor
        newval = std::max(newval, state[i*stride + j-1]);      // West neighbor
        newval = std::max(newval, state[i*stride + j+1]);      // East neighbor

        // Count changes for convergence detection
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
  // Implementation class using the PIMPL (Pointer to Implementation) pattern
  // This allows for better encapsulation and reduced compilation dependencies

  ParallelDecomp p;  // Handles domain decomposition for MPI parallelism

  // Main computational arrays - double-buffered for efficient swapping
  // Would perhaps be better to use vector or at least a unique_ptr,
  // but want to borrow pointers, swap, then set them back.
  int* state;  // Current state grid
  int* tmp;    // Temporary grid for next iteration

  // GPU related variables
  MPI_Comm node_comm;  // Communicator for processes on the same node
  int node_rank;       // Rank within the node
  int device_id;       // Assigned GPU device ID
  bool using_gpu;      // Flag indicating if GPU is being used

  // Neighbouring ranks. -1 => no neighbour
  // Ordering is NESW (North, East, South, West)
  std::array<int, 4> neigh_ranks;   // MPI ranks of neighbors
  std::array<int, 4> neigh_counts;  // Number of elements to communicate with each neighbor

  // For a simple halo exchange are going to double buffer.  Copy the
  // potentially strided data out of the array, send/recv it and then
  // copy it out on the other side.
  //
  // Allocate enough space for all 4 directions in c'tor, even if all
  // directions not needed.
  std::vector<int> halo_send_buf;  // Buffer for sending boundary data
  std::vector<int> halo_recv_buf;  // Buffer for receiving boundary data

  Impl(ParallelDecomp const& pd) : p(pd) {
    // First, create a communicator for processes on the same node
    // This allows for optimized intra-node communication and GPU assignment
    MPI_Comm_split_type(p.comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &node_comm);
    MPI_Comm_rank(node_comm, &node_rank);
    
    // Get number of MPI processes on this node
    int node_size;
    MPI_Comm_size(node_comm, &node_size);
    
    // Query available GPU devices through OpenMP
    int num_devices = omp_get_num_devices();
    
    // Set default to not using GPU
    using_gpu = false;
    device_id = -1;
    
    // Print basic information from all processes
    char hostname[256];
    gethostname(hostname, sizeof(hostname));
    
    // Gather information from all processes to print on root
    // This helps with debugging and understanding resource allocation
    struct {
      int global_rank;     // MPI rank in global communicator
      int node_rank;       // MPI rank within the node
      int num_devices;     // Number of GPU devices available
      int assigned_device; // Assigned GPU device ID
      char hostname[256];  // Node hostname
    } my_info, *all_info = nullptr;
    
    // Fill in local info
    my_info.global_rank = p.rank;
    my_info.node_rank = node_rank;
    my_info.num_devices = num_devices;
    my_info.assigned_device = -1;  // Will be set later
    strncpy(my_info.hostname, hostname, sizeof(my_info.hostname));
    
    // Allocate buffer on root process for gathering information
    int world_size;
    MPI_Comm_size(p.comm, &world_size);
    if (p.rank == 0) {
        all_info = new decltype(my_info)[world_size];
    }
    
    if (num_devices > 0) {
      // We have GPUs available
      // Assign GPU in round-robin fashion within the node (4 GPUs per node)
      // This ensures even distribution of MPI processes across available GPUs
      device_id = node_rank % num_devices;
      my_info.assigned_device = device_id;
      
      // Set this device as the default for OpenMP
      omp_set_default_device(device_id);
      
      // Test if the GPU is actually usable
      // Some GPUs might be visible but not usable due to system restrictions
      int is_device_usable = 0;
      #pragma omp target map(from:is_device_usable)
      {
        is_device_usable = 1;  // This will only execute if the device is accessible
      }
      
      if (is_device_usable) {
        using_gpu = true;
      } else {
        // Fall back to CPU if GPU is not usable
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

    // Allocate memory for state arrays
    // Size includes ghost cells (halos) for boundary exchange
    state = new int[local_size()];
    tmp = new int[local_size()];

    // Set up neighbor information for halo exchange
    // Neighbour ranks: NESW (North, East, South, West)
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

    // Allocate communication buffers for halo exchange
    // Need buffers for all four directions (N, E, S, W)
    auto halo_buf_size = 2 * (p.sub_nx + p.sub_ny);
    halo_send_buf.resize(halo_buf_size);
    halo_recv_buf.resize(halo_buf_size);
    
    // If we're using GPU, allocate device memory
    // This preallocates memory on the GPU to avoid repeated allocations
    if (using_gpu) {
      #pragma omp target enter data map(alloc: state[0:local_size()], tmp[0:local_size()])
    }
  }
    
  // Destructor - clean up resources
  ~Impl() {
    // Free GPU memory if we were using it
    if (using_gpu) {
      #pragma omp target exit data map(delete: state[0:local_size()], tmp[0:local_size()])
    }
    
    // Free host memory
    delete[] state;
    delete[] tmp;
    
    // Free the node communicator
    MPI_Comm_free(&node_comm);
  }

  // Helper function to calculate the total size of local domain including halos
  int local_size() const {
    return (p.sub_nx + 2)*(p.sub_ny + 2);  // +2 accounts for ghost cells on each dimension
  }

  // Communicate the valid edge sites to neighbours that exist. Fill
  // halos on the receiving side.
  // This function handles the critical data exchange between distributed processes:
  // 1. Device to Host: If using GPU, first copy data from device to host
  // 2. Process to Process: Exchange boundary data between MPI processes
  // 3. Host to Device: If using GPU, copy updated data back to device
  void halo_exchange(int* data) {
    // For GPU execution, copy from device to host
    // This is necessary because MPI typically cannot directly access GPU memory
    if (using_gpu) {
      #pragma omp target update from(data[0:local_size()])
    }
    
    // Create MPI requests for communication
    // We need 8 requests: 4 for receives, 4 for sends (N, E, S, W directions)
    std::array<MPI_Request, 8> reqs;  // 4 for recv, 4 for send
    auto const stride = p.sub_ny + 2;

    // Post receives first (best practice in MPI to avoid deadlocks)
    for (int b = 0, offset = 0; b < 4; ++b) {
      if (neigh_ranks[b] == -1) {
        // No neighbor in this direction (e.g., at domain boundary)
        reqs[b] = MPI_REQUEST_NULL;
      } else {
        // Post non-blocking receive for data from this neighbor
        MPI_Irecv(halo_recv_buf.data() + offset, neigh_counts[b], MPI_INT,
                 neigh_ranks[b], 0, p.comm, &reqs[b]);
      }
      offset += neigh_counts[b];
    }

    // Pack the send buffers with boundary data
    // We need to extract the edge data from our grid to send to neighbors
    int offset = 0;
    
    // N: Extract top row to send to northern neighbor
    for (int i = 1; i <= p.sub_nx; ++i) {
      halo_send_buf[offset++] = data[i * stride + p.sub_ny];
    }
    
    // E: Extract rightmost column to send to eastern neighbor
    for (int j = 1; j <= p.sub_ny; ++j) {
      halo_send_buf[offset++] = data[p.sub_nx * stride + j];
    }
    
    // S: Extract bottom row to send to southern neighbor
    for (int i = 1; i <= p.sub_nx; ++i) {
      halo_send_buf[offset++] = data[i * stride + 1];
    }
    
    // W: Extract leftmost column to send to western neighbor
    for (int j = 1; j <= p.sub_ny; ++j) {
      halo_send_buf[offset++] = data[1*stride + j];
    }

    // Post sends with non-blocking operations
    for (int b = 0, offset = 0; b < 4; ++b) {
      if (neigh_ranks[b] != -1) {
        // Send packed data to the corresponding neighbor
        MPI_Isend(halo_send_buf.data() + offset, neigh_counts[b], MPI_INT,
                 neigh_ranks[b], 0, p.comm, &reqs[b+4]);
      } else {
        // No neighbor in this direction
        reqs[b+4] = MPI_REQUEST_NULL;
      }
      offset += neigh_counts[b];
    }
    
    // Wait for all communications to complete
    // This ensures all data has been exchanged before continuing
    MPI_Waitall(8, reqs.data(), MPI_STATUSES_IGNORE);

    // Unpack the received data into the ghost cells
    offset = 0;
    
    // N: Fill ghost cells above our domain with data from northern neighbor
    for (int i = 1; i <= p.sub_nx; ++i) {
      data[i * stride + p.sub_ny + 1] = halo_recv_buf[offset++];
    }
    
    // E: Fill ghost cells to the right with data from eastern neighbor
    for (int j = 1; j <= p.sub_ny; ++j) {
      data[(p.sub_nx + 1) * stride + j] = halo_recv_buf[offset++];
    }
    
    // S: Fill ghost cells below with data from southern neighbor
    for (int i = 1; i <= p.sub_nx; ++i) {
      data[i * stride + 0] = halo_recv_buf[offset++];
    }
    
    // W: Fill ghost cells to the left with data from western neighbor
    for (int j = 1; j <= p.sub_ny; ++j) {
      data[0*stride + j] = halo_recv_buf[offset++];
    }
    
    // For GPU execution, update device with new data
    // After halo exchange, we need to update the GPU memory with the new ghost cells
    if (using_gpu) {
      #pragma omp target update to(data[0:local_size()])
    }
  }
};

// Constructor - creates the implementation object
GpuRunner::GpuRunner(ParallelDecomp const& p) : m_impl(std::make_unique<Impl>(p)) {
}

// Destructor - automatically cleans up implementation through unique_ptr
GpuRunner::~GpuRunner() = default;

// Give each process a sub-array of size (sub_nx+2) x (sub_ny+2)
// Copies input data from source to the local state array
void GpuRunner::copy_in(int const* source) {
  // Copy data from source to local state array
  std::copy(source, source + m_impl->local_size(), m_impl->state);
  
  // For GPU execution, copy data to device
  // This ensures the GPU has the latest data before computation
  if (m_impl->using_gpu) {
    #pragma omp target update to(m_impl->state[0:m_impl->local_size()])
  }
}

// Give each process a sub-array of size (sub_nx+2) x (sub_ny+2)
// Copies computed results from local state to the output destination
void GpuRunner::copy_out(int* dest) const {
  // For GPU execution, copy data from device
  // We need to get the final results from GPU memory
  if (m_impl->using_gpu) {
    #pragma omp target update from(m_impl->state[0:m_impl->local_size()])
  }
  
  // Copy data from local state to destination
  std::copy(m_impl->state, m_impl->state + m_impl->local_size(), dest);
}

// Iteratively perform percolation of the non-zero elements until no
// changes or 4 *max(M, N) iterations.
// This is the main computational loop that drives the simulation
void GpuRunner::run() {
  auto& p = m_impl->p;

  int const M = p.nx;
  int const N = p.ny;
  
  // Timing variables for performance analysis
  double start_time = MPI_Wtime();
  double compute_time = 0.0;
  double comm_time = 0.0;
  double total_time;
  
  // Log GPU utilization at the start
  // This helps understand how many processes are actually using GPUs
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
  
  // For GPU execution, update device memory with initial tmp array
  if (m_impl->using_gpu) {
    #pragma omp target update to(m_impl->tmp[0:m_impl->local_size()])
  }

  // Setup iteration parameters
  int const maxstep = 4 * std::max(M, N);  // Maximum iteration limit based on grid size
  int step = 1;
  int global_nchange = 1;  // Start with a non-zero value to enter the loop

  // Use pointers to the buffers (which we swap below) to avoid copies.
  // This implements double-buffering for efficient iteration
  int* current = m_impl->state;
  int* next = m_impl->tmp;

  // Main iteration loop - continues until convergence or max steps reached
  while (global_nchange && step <= maxstep) {
    // Ensure edge sites have been communicated to neighbouring
    // processes.
    double comm_start = MPI_Wtime();
    m_impl->halo_exchange(current);
    comm_time += MPI_Wtime() - comm_start;
    
    // Update this process's subdomain - GPU or CPU implementation
    double compute_start = MPI_Wtime();
    int local_nchange;
    
    if (m_impl->using_gpu) {
      // Use GPU implementation for computation
      local_nchange = percolate_gpu_step(p.sub_nx, p.sub_ny, current, next);
    } else {
      // Use CPU implementation if GPU is not available
      local_nchange = 0;
      int const stride = p.sub_ny + 2;
      
      // CPU version of the percolation algorithm
      for (int i = 1; i <= p.sub_nx; ++i) {
        for (int j = 1; j <= p.sub_ny; ++j) {
          int const idx = i*stride + j;
          int const oldval = current[idx];
          int newval = oldval;

          // 0 => solid, so do nothing
          if (oldval != 0) {
            // Set next[i][j] to be the maximum value of state[i][j] and
            // its four nearest neighbours
            newval = std::max(newval, current[(i-1)*stride + j  ]);  // North
            newval = std::max(newval, current[(i+1)*stride + j  ]);  // South
            newval = std::max(newval, current[    i*stride + j-1]);  // West
            newval = std::max(newval, current[    i*stride + j+1]);  // East

            if (newval != oldval) {
              ++local_nchange;
            }
          }

          next[idx] = newval;
        }
      }
    }
    compute_time += MPI_Wtime() - compute_start;

    // Share the total changes to make a decision on convergence
    // Need to aggregate changes across all processes
    comm_start = MPI_Wtime();
    MPI_Allreduce(&local_nchange,
         &global_nchange,
         1, MPI_INT, MPI_SUM, p.comm);
    comm_time += MPI_Wtime() - comm_start;

    // Report progress every now and then
    if (step % printfreq == 0) {
      p.print_root("percolate: number of changes on step {} is {}\n",
         step, global_nchange);
    }

    // Swap the pointers for the next iteration
    // This is more efficient than copying arrays
    std::swap(next, current);
    step++;
  }

  // Answer now in `current`, make sure this one is in `state`
  // Ensure the final result is in the state array, not the temporary
  if (current != m_impl->state) {
    // Copy data from current to state
    std::memcpy(m_impl->state, current, sizeof(int) * m_impl->local_size());
    
    // For GPU execution, ensure state is updated on device
    if (m_impl->using_gpu) {
      #pragma omp target update to(m_impl->state[0:m_impl->local_size()])
    }
  }
  
  // Update pointers to maintain a consistent state
  m_impl->state = current;
  m_impl->tmp = next;
  
  // Calculate total time
  total_time = MPI_Wtime() - start_time;
  
  // Gather and report timing statistics
  // This is crucial for performance analysis
  if (p.rank == 0) {
    printf("\n==== Percolation Complete ====\n");
    printf("Completed in %d steps (max allowed: %d)\n", step, maxstep);
    printf("Total runtime: %.3f seconds\n", total_time);
    printf("Computation time: %.3f seconds (%.1f%%)\n", 
           compute_time, (compute_time/total_time)*100.0);
    printf("Communication time: %.3f seconds (%.1f%%)\n", 
           comm_time, (comm_time/total_time)*100.0);
    printf("============================\n\n");
  }
}