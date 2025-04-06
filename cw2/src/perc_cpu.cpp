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

#include "perc_cpu.h"

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
static int percolate_cpu_step(int M, int N, int const* state, int* next) {
  int nchange = 0;
  int const stride = N + 2;

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

      next[(i)*(N+2) + j] = newval;
    }
  }
  return nchange;
}


struct CpuRunner::Impl {
  // Here you can store any parameters or data needed for
  // implementation of your version.

  ParallelDecomp p;

  // Would perhaps be better to use vector or at least a unique_ptr,
  // but want to borrow pointers, swap, then set them back.
  int* state;
  int* tmp;

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
  }
    
  ~Impl() {
    delete[] state;
    delete[] tmp;
  }

  int local_size() const {
    return (p.sub_nx + 2)*(p.sub_ny + 2);
  }

  // Communicate the valid edge sites to neighbours that exist. Fill
  // halos on the receiving side.
  void halo_exchange(int* data) {
    // Recall: order of directions is NESW.
    // 
    // Strategy: post non-blocking receives into recv buf, pack send
    // buf, send send buf, wait, unpack recv buf.
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
  }
};

CpuRunner::CpuRunner(ParallelDecomp const& p) : m_impl(std::make_unique<Impl>(p)) {
}

CpuRunner::~CpuRunner() = default;

// Give each process a sub-array of size (sub_nx+2) x (sub_ny+2)
void CpuRunner::copy_in(int const* source) {
  std::copy(source, source + m_impl->local_size(), m_impl->state);
}

// Give each process a sub-array of size (sub_nx+2) x (sub_ny+2)
void CpuRunner::copy_out(int* dest) const {
  std::copy(m_impl->state, m_impl->state + m_impl->local_size(), dest);
}

// Iteratively perform percolation of the non-zero elements until no
// changes or 4 *max(M, N) iterations.
void CpuRunner::run() {
  auto& p = m_impl->p;

  int const M = p.nx;
  int const N = p.ny;
  // Copy the initial state to the temp, only the global halos are
  // *required*, but much easier this way!
  std::memcpy(m_impl->tmp, m_impl->state, sizeof(int) * m_impl->local_size());

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
    int local_nchange = percolate_cpu_step(p.sub_nx, p.sub_ny, current, next);

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
  m_impl->state = current;
  m_impl->tmp = next;
}
