// -*- mode: C++; -*-
// Copyright (C) 2025, Rupert Nash, The University of Edinburgh.
//
// All rights reserved.
//
// This file is provided to you to complete an assessment and for
// subsequent private study. It may not be shared and, in particular,
// may not be posted on the internet. Sharing this or any modified
// version may constitute academic misconduct under the University's
// regulations.

#ifndef ASPP_CUDACW_DECOMP_H
#define ASPP_CUDACW_DECOMP_H

#include <cstdio>
#include <utility>

#include <mpi.h>
#include <fmt/base.h>

struct ParallelDecomp {
  MPI_Comm comm;
  int rank;
  int nproc;

  // Total domain size
  int nx;
  int ny;

  // Number of processes on each dimension
  int px = 1;
  int py = 1;
  // Index of this process in each dimension
  int pi = 0;
  int pj = 0;
  // Size of the per-process subdomain (well, the max size as the last
  // in each dim can be smaller)
  int sub_nx;
  int sub_ny;

  ParallelDecomp(MPI_Comm c, int NX, int NY, int PX, int PY);

  inline int global_rank(int ri, int rj) {
    return ri*py + rj;
  }

  // Overload for stdout
  template <typename... T>
  void print_root(char const* msg, T&&... args) {
    if (rank == 0)
      fmt::print(msg, std::forward<T>(args)...);
  }

  // Overload to print to a file
  template <typename... T>
  void print_root(std::FILE* stream, char const* msg, T&&... args) {
    if (rank == 0)
      fmt::print(stream, msg, std::forward<T>(args)...);
  }
};

#endif
