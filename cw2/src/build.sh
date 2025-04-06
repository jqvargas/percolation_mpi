#!/bin/bash

# Build script for percolation simulation with OpenMP offloading
# This script builds the project using CMake with OpenMP target offloading for GPU acceleration

# Exit on error
set -e

# Print commands as they are executed
set -x

# Load required modules
echo "Loading required modules..."
module load gcc/10.2.0 
module load nvidia/nvhpc-nompi/24.5 
module load openmpi/4.1.6-cuda-12.4
module load cmake 

# Set environment variables for GPU visibility
export NVCOMPILER_ACC_GPU_TARGET=cc70
export NVCOMPILER_CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda
export CUDA_HOME=/opt/nvidia/hpc_sdk/Linux_x86_64/24.5/cuda

# Set MPI environment variables to help CMake find MPI
export MPI_C_COMPILER=$(which mpicc)
export MPI_CXX_COMPILER=$(which mpicxx)
export OMPI_CXX=nvc++

# Create build directory if it doesn't exist
mkdir -p build

# Configure with OpenMP
echo "Configuring CMake with OpenMP offloading..."
cmake -S . -B build \
    -DACC_MODEL=OpenMP \
    -DCMAKE_CXX_COMPILER=nvc++ \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DMPI_CXX_COMPILER=$(which mpicxx) \
    -DMPI_C_COMPILER=$(which mpicc)

# Build the project
echo "Building the project..."
cmake --build build -j 4

# Verify the executable exists
if [ -f "build/test" ]; then
    echo "Build completed successfully!"
    echo "The executable 'test' has been created in the build directory."
    echo ""
    echo "To run the program with GPU support, use:"
    echo "1. For single node (4 GPUs): mpirun -np 4 ./build/test"
    echo "2. For two nodes (8 GPUs): srun -n 8 --gpus-per-node=4 ./build/test"
    echo ""
    echo "Note: For SLURM jobs, add these directives to your submission script:"
    echo "#SBATCH --nodes=2"
    echo "#SBATCH --ntasks-per-node=4"
    echo "#SBATCH --gpus-per-node=4"
else
    echo "Error: Executable not found in build directory!"
    exit 1
fi 