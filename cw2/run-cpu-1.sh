#!/bin/bash --login
#SBATCH --partition=standard
#SBATCH --qos=short
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:05:00

module load gcc/10.2.0 nvidia/nvhpc-nompi/24.5 openmpi/4.1.6-cuda-12.4

# SYCL
module load oneapi
module load compiler

# OpenMP
export SRUN_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "Small test"
srun --cpu-bind=cores build/test -s 987 -M 512 -N 512 -P 1 -Q 1 -o small-$SLURM_JOB_ID.png

echo "Medium test"
# reduce porosity to reduce run time
srun --cpu-bind=cores build/test -s 9876 -p 0.5 -M 1024 -N 2048 -P 1 -Q 1 -o medium-$SLURM_JOB_ID.png

echo "Large test"
srun --cpu-bind=cores build/test -s 98765 -p 0.3 -M 4096 -N 4096 -P 1 -Q 1 -o large-$SLURM_JOB_ID.png
