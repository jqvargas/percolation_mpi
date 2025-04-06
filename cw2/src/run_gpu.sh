#!/bin/bash
#SBATCH --job-name=percolation_gpu
#SBATCH --time=01:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --partition=gpu
#SBATCH --account=<your_cirrus_budget_code>

# Print basic job information
echo "Running on nodes: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes with $SLURM_NTASKS total tasks"
echo "GPUs per node: $SLURM_GPUS_PER_NODE"

# Load required modules exactly as specified in the instructions
module load cmake gcc/10.2.0 nvidia/nvhpc-nompi/24.5 openmpi/4.1.6-cuda-12.4

# Display GPU information
nvidia-smi

# Run the executable with SLURM
srun -n 8 --gpus-per-node=4 ./build/test

# Alternative run using MPI directly (sometimes better for GPU binding)
# mpirun -n 8 --map-by ppr:4:node:pe=1 --bind-to core --report-bindings ./build/test 