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

# Create build directory if it doesn't exist
mkdir -p build

# Configure with OpenMP
echo "Configuring CMake with OpenMP offloading..."
cmake -S . -B build \
    -DACC_MODEL=OpenMP \
    -DCMAKE_CXX_COMPILER=nvc++ \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Build the project
echo "Building the project..."
cmake --build build -j 4

# Verify the executable exists
if [ -f "build/test" ]; then
    echo "Build completed successfully!"
    echo "The executable 'test' has been created in the build directory."
    echo ""
    echo "To run the program:"
    echo "1. For single GPU: mpirun -np 1 ./build/test"
    echo "2. For multiple GPUs: mpirun -np 4 ./build/test"
else
    echo "Error: Executable not found in build directory!"
    exit 1
fi 