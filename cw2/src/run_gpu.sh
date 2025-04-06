#!/bin/bash
#SBATCH --job-name=percolation_gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --partition=gpu
#SBATCH --account=<your_cirrus_budget_code>

# Print info about the job
echo "Running on nodes: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes with $SLURM_NTASKS total tasks"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"

# Load required modules
module load gcc/10.2.0 
module load nvidia/nvhpc-nompi/24.5 
module load openmpi/4.1.6-cuda-12.4

# Set critical environment variables for GPU visibility
export NVCOMPILER_ACC_GPU_TARGET=cc70
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Display GPU information
nvidia-smi

# Run the executable with SLURM - this ensures proper GPU binding
srun -n 8 --gpus-per-node=4 ./build/test

# Alternative run using MPI directly (sometimes better for GPU binding)
# mpirun -n 8 --map-by ppr:4:node:pe=1 --bind-to core --report-bindings ./build/test 