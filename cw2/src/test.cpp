// Copyright (C) 2025, Rupert Nash, The University of Edinburgh.
//
// All rights reserved.
//
// This file is provided to you to complete an assessment and for
// subsequent private study. It may not be shared and, in particular,
// may not be posted on the internet. Sharing this or any modified
// version may constitute academic misconduct under the University's
// regulations.

#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <numeric>
#include <vector>

#include <mpi.h>
#include <fmt/base.h>

#include "decomp.h"
#include "util.h"
#include "perc_cpu.h"
#include "perc_gpu.h"

using Clock = std::chrono::high_resolution_clock;
using Time = Clock::time_point;
using Dur = Clock::duration;

char const* usage =
  "Benchmark percolation implementation\n"
  "    test [-M integer] [-N integer] [-P integer] [-Q integer] [-s integer] [-r integer] [-p float] [-o filename]\n"
  "\n"
  "-M grid size in x direction\n"
  "-N grid size in y direction\n"
  "-P number of MPI processes in x direction\n"
  "-Q number of MPI processes in y direction\n"
  "-s random seed\n"
  "-r number of repeats for benchmarking\n"
  "-p target porosity\n"
  "-o file name to write output PNG image\n"
  ;

int main(int argc, char* argv[]) {

  int seed = 1234;
  int M = 512;
  int N = 512;
  int P = 1;
  int Q = 1;
  float porosity = 0.6;
  int nruns = 3;
  char const* img_fn = "test.png";

  // Technically must do this before accessing command line arguments.
  MPI_Init(&argc, &argv);
  // We'll not install an MPI error handler, so it remains
  // MPI_ERRORS_ARE_FATAL, so no point checking return codes.

  for (int i = 1; i < argc; i += 2) {
    char const* flag = argv[i];
    char const* value = argv[i + 1];
    if (std::strncmp("-M", flag, 2) == 0) {
      M = std::atoi(value);
    } else if (std::strncmp("-N", flag, 2) == 0) {
      N = std::atoi(value);
    } else if (std::strncmp("-s", flag, 2) == 0) {
      seed = std::atoi(value);
    } else if (std::strncmp("-p", flag, 2) == 0) {
      porosity = std::atof(value);
    } else if (std::strncmp("-r", flag, 2) == 0) {
      nruns = std::atoi(value);
    } else if (std::strncmp("-o", flag, 2) == 0) {
      img_fn = value;
    } else if (std::strncmp("-P", flag, 2) == 0) {
      P = std::atoi(value);
    } else if (std::strncmp("-Q", flag, 2) == 0) {
      Q = std::atoi(value);
    } else {
      fmt::print(stderr, "Unknown flag: '{}'\n{}", flag, usage);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  auto par = ParallelDecomp(MPI_COMM_WORLD, M, N, P, Q);

  par.print_root("Total number of processes: {}\n", par.nproc);
  par.print_root("Process decomposition: P = {}, Q = {}\n", P, Q);
  if (P * Q != par.nproc) {
    par.print_root(stderr, "Process number mismatch!\n");
    MPI_Abort(par.comm, 1);
  }
    
  par.print_root("Total number of sites: M = {}, N = {}\n", M, N);
  par.print_root("Per-process number of sites: {}, {}\n", par.sub_nx, par.sub_ny);

  // Allocate data for the max number of per-process sites, even if
  // this process doesn't need them all.
  std::vector<int> map((par.sub_nx + 2) * (par.sub_ny + 2));
  int proc_hole = fill_map(seed, porosity, &par, map.data());
  int total_hole = 0;
  MPI_Allreduce(&proc_hole, &total_hole, 1, MPI_INT, MPI_SUM, par.comm);

  par.print_root("Porosity: target = {:f}, actual = {:f}\n",
		 porosity, ((double) total_hole)/((double) M*N) );

  auto benchmarker = [&](
		   int nruns,
		   std::vector<int> const& init,
		   std::vector<double>& time_s,
		   auto& runner
		   ) {
    par.print_root("Starting {} runs\n", nruns);
    for (int i = 0; i < nruns; ++i) {
      runner.copy_in(init.data());
      Time const start = Clock::now();
      runner.run();
      Time const stop = Clock::now();
      std::chrono::duration<double> dt{stop - start};
      time_s[i] = dt.count();
      par.print_root("Run {}, time = {:f} s\n", i, dt.count());
    }
  };

  std::vector<double> cpu_time_s(1);
  std::vector<int> cpu_state(map.size());
  std::vector<double> gpu_time_s(nruns);
  std::vector<int> gpu_state(map.size());

  {
    par.print_root("CPU section\n");
    auto cpu_r = CpuRunner(par);
    benchmarker(1, map, cpu_time_s, cpu_r);
    cpu_r.copy_out(cpu_state.data());
  
    par.print_root("GPU section\n");
    auto gpu_r = GpuRunner(par);
    benchmarker(nruns, map, gpu_time_s, gpu_r);
    gpu_r.copy_out(gpu_state.data());
  }

  // Check for match
  int const local_ndiff =
    std::inner_product(
      cpu_state.begin(), cpu_state.end(),
      gpu_state.begin(),
      0,
      std::plus<int>{},
      [](int const& a, int const& b) {
        return a == b ? 0 : 1;
      }
    );

  int global_ndiff;
  MPI_Allreduce(&local_ndiff, &global_ndiff, 1, MPI_INT, MPI_SUM, par.comm);
  
  if (global_ndiff) {
    par.print_root("CPU and GPU results differ at {} locations\n", global_ndiff);
    return 1;
  }

  par.print_root("CPU and GPU results match\n");

  auto print_stats = [&] (std::vector<double> const& data, char const* where) {
    // Compute and print stats
    int N = data.size();
    double min = INFINITY;
    double max = -INFINITY;
    double tsum = 0.0, tsumsq = 0.0;
    for (int i = 0; i < N; ++i) {
      double const& t = data[i];
      tsum += t;
      tsumsq += t * t;
      min = (t < min) ? t : min;
      max = (t > max) ? t : max;
    }
    double mean = tsum / N;
    double tvar = (tsumsq - tsum*tsum / N) / (N - 1);
    double std = std::sqrt(tvar);
    par.print_root("\nSummary for {} (all in s):\nmin = {:e}, max = {:e}, mean = {:e}, std = {:e}\n",
		where,
		min, max, mean, std);
  };

  print_stats(cpu_time_s, "CPU");
  print_stats(gpu_time_s, "GPU");

  par.print_root("Writing image to '{}'\n", img_fn);
  write_state_png(img_fn, &par, cpu_state.data());

  MPI_Finalize();
}
