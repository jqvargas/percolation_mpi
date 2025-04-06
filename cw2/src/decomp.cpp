// Copyright (C) 2025, Rupert Nash, The University of Edinburgh.
//
// All rights reserved.
//
// This file is provided to you to complete an assessment and for
// subsequent private study. It may not be shared and, in particular,
// may not be posted on the internet. Sharing this or any modified
// version may constitute academic misconduct under the University's
// regulations.

#include "decomp.h"

#include <fmt/base.h>

ParallelDecomp::ParallelDecomp(
    MPI_Comm c, int NX, int NY, int PX, int PY
) : comm(c), nx(NX), ny(NY), px(PX), py(PY) {
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nproc);
  if (PX * PY != nproc) {
    fmt::print(stderr, "Parallel decomposition error: {} x {} != {}",
	       PX, PY, nproc);
  }

  pi = rank / PY;
  pj = rank % PY;

  if (global_rank(pi, pj) != rank) {
    fmt::print(stderr, "Rank error");
    MPI_Abort(comm, 1);
  }

  // Integer ceiling division for number of sites per process
  sub_nx = (NX - 1)/PX + 1;
  sub_ny = (NY - 1)/PY + 1;

}
