// Copyright (C) 2025, Rupert Nash, The University of Edinburgh.
//
// All rights reserved.
//
// This file is provided to you to complete an assessment and for
// subsequent private study. It may not be shared and, in particular,
// may not be posted on the internet. Sharing this or any modified
// version may constitute academic misconduct under the University's
// regulations.

#ifndef ASPP_CUDACW_PERC_CPU_H
#define ASPP_CUDACW_PERC_CPU_H

#include <memory>

struct ParallelDecomp;

struct CpuRunner {
  struct Impl;
  std::unique_ptr<Impl> m_impl;

  // Create state needed for this problem size
  CpuRunner(ParallelDecomp const& p);
  ~CpuRunner();

  // Fill state with input data (assuming source has size (M+2)*(N+2)
  // including the halo of zeros)
  void copy_in(int const* source);
  // Fill dest with state
  void copy_out(int* dest) const;

  // Iteratively perform percolation of the non-zero elements until no
  // changes or 4 *max(M, N) iterations.
  void run();
};

#endif
