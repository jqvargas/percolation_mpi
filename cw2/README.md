# ASPP coursework 2

Please see Learn for submission deadlines.

Remember what you submit must be your own work. Anything not your own
work which you have accessed should be correctly referenced and
cited. You must not share this assessment's source code nor your
solutions. Use of generative AI tools, in line with School policy, is
not permitted.

Please see further information and guidance from the School of
Informatics Academic Misconduct Officer:
<https://web.inf.ed.ac.uk/infweb/admin/policies/academic-misconduct>

## Summary

Your task is to take a CPU only, MPI parallel version of the
percolation model used in the last coursework and make this run
correctly and efficiently on multiple Cirrus nodes, on both the CPU
and GPU partitions. To do this, you will need to use OpenMP offloading
or SYCL.

You will have to prepare a brief report about this, justifying your
choices and comparing the performance achieved.

## Problem description

So called "percolation models" are used as simplified models of
various systems, including forest fires, disease propagation, and flow
in porous media. Here we focus on the latter, asking: given a random
lattice 2D material with a porosity `p` (i.e. the material is
approximated as a grid of equal sized squares, each being empty with
independent probability `p`), do the pore spaces connect?

This can be done by first labelling each non-solid cell with a unique
number, then iteratively updating this value to the maximum of the
labels at that cell and its four immediate neighbours, i.e.

```
new_label[i,j] = max(
                label[  i,j+1],
label[i-1,  j], label[  i,  j], label[i+1, j]
                label[  i,j-1]
)
```

To know when to finish, the algorithm needs to know when this process
has converged, i.e. when the total number of changes across the whole
grid is zero.

The supplied code applies this, in parallel with MPI, on the CPU. The
driver code runs this once on the CPU to compute a reference solution,
then a number of times using the GPU implementation (in the initial
code, this is simply a copy of the CPU implementation with an altered
name), before validating the solution, printing timing statistics, and
writing the a image of the output.

![Sample output image](default_output.png)

It accepts the following flags:

`-M <integer>` - horizontal size of grid (default 512)

`-N <integer>` - vertical size of grid (default 512)

`-P <integer>` - number of processes in the horizonal direction (default 1)

`-Q <integer>` - number of processes in the vertical direction (default 1)

`-s <integer>` - random seed (default 1234)

`-r <integer>` - number of repeats for benchmarking (default 3)

`-p <float>` - target porosity between 0 and 1 (default 0.6)

`-o <path>` - file name to write output PNG image (default `test.png`)

## Set up
On Cirrus:

1. Clone the code repository (recalling that only the work filesystem is accessible from the compute nodes): `git clone /home/z04/shared/aspp/cw2`

2. Load the required modules (note use of the NVHPC toolkit without
   MPI and the alternative module):
```bash
module load cmake gcc/10.2.0 \
	nvidia/nvhpc-nompi/24.5 openmpi/4.1.6-cuda-12.4
# Only required for SYCL
module load oneapi # access Intel OneAPI module collection
module load compiler # load Intel SYCL compiler
```

3. Configure the code. The supplied CMake accepts a flag `ACC_MODEL`
   to specify which programming model wish to use. Choose from `None`,
   `SYCL`, `OpenMP`, and `CUDA`. For final runs, I suggest you use a
   build type of `Release`, but you may prefer to build with some
   debug info while testing.

```bash
cmake -S src -B build-cuda \
  -DACC_MODEL=CUDA \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo
```

4. Compile: `cmake --build build -j 4`. This will produce an
   executable `test`.

5. Running the *unmodified* code is possible on the login nodes. If
   you run with no options (just `build/test`) the produced image
   should be identical to the `default_output.png` in the root of the
   repository.
   
6. I have provided several template batch scripts that run on
   different numbers of CPU and GPUs (`run-gpu-1.sh` etc).

## Requirements for your code

You must adapt the code so that it runs on (at a minimum):
 - 2 Cirrus GPU nodes (i.e. 8 x V100)
 - 4 Cirrus CPU nodes (i.e. 8 x 18-core Xeon)
correctly and with the best performance you can manage.

To constrain the scope of the task, as before you are only allowed to
modify the GPU implementation `perc_gpu.cpp`. This file must be
submitted with your report.

You should use *one* of SYCL *or* OpenMP for your implementation. If
you use SYCL, you should using the Intel OneAPI DPC++ compiler
(`icpx`). If you use OpenMP, you should use the NVIDIA HPC C++
compiler (`nvc++`). When configuring with CMake, make sure you set the
`ACC_MODEL` variable appropriately and also `CMAKE_CXX_COMPILER`.

You may wish to change the behaviour of the code depending on where
you are using CPU or GPU. Both models have features to do this.

SYCL: query the device for the "gpu" or "cpu" aspect
```C++
sycl::queue Q = whatever();
auto dev = Q.get_device();
if (dev.has<sycl::aspect::gpu>()) {
  // Do GPU things
} else {
  // Do CPU things
}
```

OpenMP: query the number of non-host devices or use metadirectives
```C++
int ndev = omp_get_num_devices();
if (ndev == 0) {
  // Running on CPU as only have host device
} else {
  // Running on GPU
}

#pragma omp metadirective \
when(device={kind("gpu")}: target teams distribute) \
otherwise(parallel for)
for (int i =0; i < N; ++i) {
 // whatever
}

```

## Requirements for report

You must submit a report (maximum 8 pages) in PDF format with the
following sections.

1. Build instructions

Give instructions on how to configure, compile, and run your version
of the application. Assume that the reader starts with a clean
checkout of the repository, replacing `perc_gpu.cpp` with your
submitted file. You may wish to set CMake options etc. You may provide
different instructions for CPU and GPU versions but the source code
must be the same.

2. Design

Which programming model have you chosen and why? Be sure here to make
reference to the algorithm and its implementation to justify your
choice.

Briefly explain the approach you have taken to developing your code
and the reasons for these choices. You may wish to include
profiling/timing results from interim versions of your code. You
should address both single-device performance as well as steps
relating to message passing.

3. Performance results

Show performance results for a range of problem sizes and numbers of
MPI processes. Sizes should range from 512 to 4096 (inclusive) and MPI
processes from 1 to 8 (i.e. one MPI process per compute device, either
V100 or Xeon). You should include figures.

4. Discussion

Here you should reflect on the process you used and how well you did
in terms of reaching the best achievable performance. This should
include both optimisation of the single-device performance and the
message passing aspects. Ensure you discuss how close you are to
reaching the theoretical performance of the systems. For further
credit, you might compare performance to an implementation which uses
CUDA-specific features, such as one based upon your code from the
previous coursework.

## Submission

Please see the instructions on Learn for full details, but you will
have to submit the code and report to separate queues.

**It is VITAL that you use your exam number for both the code and
report filenames so we can match them up!** I.e., the report should be
`B123456.pdf` and the code `B123456.cpp`.

## Hints

You have been provided the code in a version controlled Git
repository. I suggest that you make use of it to track your progress
and you will be able to tell if you have inadvertently modified files
that you should not with `git status`.

You have implemented a single-GPU CUDA version of this same
application. How do the versions' performance compare?

You will have to assign devices to MPI processes. The MPI function
`MPI_Comm_split_type` with a split type of `MPI_COMM_TYPE_SHARED` can
be useful here.

While you can submit only one version of the code, there is nothing to
stop you discussing any other (partial) implementations to justify
choices.
