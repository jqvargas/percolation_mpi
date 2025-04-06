# Copyright (C) 2025, Rupert Nash, The University of Edinburgh.

# All rights reserved.

# This file is provided to you to complete an assessment and for
# subsequent private study. It may not be shared and, in particular,
# may not be posted on the internet. Sharing this or any modified
# version may constitute academic misconduct under the University's
# regulations.

cmake_minimum_required(VERSION 3.18)
project(perc CXX)

include(FetchContent)

# Simplify appending to target and source properties
function(append_target_prop tgt pname)
  get_target_property(pval ${tgt} ${pname})
  list(APPEND pval ${ARGN})
  set_target_properties(${tgt} PROPERTIES ${pname} "${pval}")
endfunction()
function(append_source_prop source pname)
  get_source_file_property(pval ${source} ${pname})
  list(APPEND pval ${ARGN})
  set_source_files_properties(${source} PROPERTIES ${pname} "${pval}")
endfunction()

set(ACC_MODEL "None" CACHE STRING
    "Which programming model to use: None, OpenMP, SYCL, CUDA")

set(CMAKE_CXX_STANDARD 17)

if (ACC_MODEL STREQUAL None)
  # No per-process acceleration. Not much use!
  message(WARNING "No accelerator programming model selected! Set ACC_MODEL!")

elseif (ACC_MODEL STREQUAL OpenMP)
  # OpenMP target offloading
  if (NOT CMAKE_CXX_COMPILER_ID STREQUAL NVHPC)
    message(WARNING "Not using NVIDIA C++ compiler! May need to re-configure with CMAKE_CXX_COMPILER=nvc++")
  endif()
  # Not really needed
  find_package(OpenMP REQUIRED)

elseif (ACC_MODEL STREQUAL SYCL)
  # SYCL
  if (NOT CMAKE_CXX_COMPILER_ID STREQUAL IntelLLVM)
    message(WARNING "Not using Intel DPC++ compiler! May need to reconfigure with CMAKE_CXX_COMPILER=icpx")
  endif()
  find_package(IntelSYCL REQUIRED)

  # Set up for Cirrus: generate code for V100 GPU and multicore CPU
  set(SYCL_TARGET_FLAGS "-fsycl-targets=spir64_x86_64,nvptx64-nvidia-cuda;-Xsycl-target-backend=nvptx64-nvidia-cuda;--offload-arch=sm_70" CACHE STRING "Flags to control which targets get code generated")
  # Need to use these when compiling and linking
  append_target_prop(IntelSYCL::SYCL_CXX INTERFACE_COMPILE_OPTIONS ${SYCL_TARGET_FLAGS})
  append_target_prop(IntelSYCL::SYCL_CXX INTERFACE_LINK_OPTIONS ${SYCL_TARGET_FLAGS})

elseif (ACC_MODEL STREQUAL CUDA)
  # This can't be used for your "real" program, but perhaps could be
  # useful when estimating how fast it is compared to a
  # vendor-specific implementation...
  message(WARNING "Configuring in CUDA mode - only useful for comparisons")

# Default for Cirrus
  set(CMAKE_CUDA_ARCHITECTURES 70-real)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  list(APPEND CMAKE_CUDA_FLAGS --expt-relaxed-constexpr)

else()
  message(FATAL_ERROR "Invalid value of ACC_MODEL = '${ACC_MODEL}'")

endif()

find_package(MPI REQUIRED)
find_package(PNG REQUIRED)

FetchContent_Declare(
  fmt
  GIT_REPOSITORY https://github.com/fmtlib/fmt
  GIT_TAG        123913715afeb8a437e6388b4473fcc4753e1c9a # 11.1.4
)
FetchContent_MakeAvailable(fmt)

add_executable(test util.cpp perc_cpu.cpp decomp.cpp perc_gpu.cpp test.cpp)

target_link_libraries(test PRIVATE MPI::MPI_CXX PNG::PNG fmt::fmt)

if (ACC_MODEL STREQUAL None)
  # Nothing to do

elseif (ACC_MODEL STREQUAL OpenMP)
  # Generate device code for V100 and host
  append_source_prop(perc_gpu.cpp COMPILE_OPTIONS -mp=gpu -gpu=cuda12.4,cc70 -Minfo=mp)
  target_link_options(test PRIVATE -mp=gpu)

elseif (ACC_MODEL STREQUAL SYCL)
  add_sycl_to_target(TARGET test SOURCES perc_gpu.cpp)

elseif (ACC_MODEL STREQUAL CUDA)
  set_source_files_properties(perc_gpu.cpp PROPERTIES LANGUAGE CUDA)
  set_target_properties(test PROPERTIES LINKER_LANGUAGE CUDA)
  set_target_properties(test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

endif()

install(TARGETS test)
