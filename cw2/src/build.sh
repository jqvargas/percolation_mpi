#!/bin/bash

# Exit on error
set -e

# Print commands as they are executed
set -x

# Load required modules exactly as specified in the instructions
module load cmake gcc/10.2.0 nvidia/nvhpc-nompi/24.5 openmpi/4.1.6-cuda-12.4

# Create build directory if it doesn't exist
mkdir -p build

# Configure with OpenMP
cmake -S . -B build \
    -DACC_MODEL=OpenMP \
    -DCMAKE_CXX_COMPILER=nvc++ \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo

# Build the project
cmake --build build -j 4

# Verify the executable exists
if [ -f "build/test" ]; then
    echo "Build completed successfully!"
else
    echo "Error: Executable not found in build directory!"
    exit 1
fi 