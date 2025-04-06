#!/bin/bash --login
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:4
#SBATCH --nodes=2
#SBATCH --exclusive
#SBATCH --time=00:05:00
#SBATCH --account=mdisspt-s2266011
module load gcc/10.2.0 nvidia/nvhpc-nompi/24.5 openmpi/4.1.6-cuda-12.4

# SYCL
module load oneapi
module load compiler

# OpenMP
export OMP_TARGET_OFFLOAD=MANDATORY

#echo "Small test"
#srun --ntasks 8 --tasks-per-node=4 build/test -s 987 -M 512 -N 512 -P 2 -Q 4 -o small-$SLURM_JOB_ID.png

#echo "Medium test"
# reduce porosity to reduce run time
#srun --ntasks 8 --tasks-per-node=4 build/test -s 9876 -p 0.5 -M 1024 -N 2048 -P 2 -Q 4 -o medium-$SLURM_JOB_ID.png

echo "Large test"
srun --ntasks 8 --tasks-per-node=4 src/build/test -s 98765 -p 0.3 -M 4096 -N 4096 -P 2 -Q 4 -o large-$SLURM_JOB_ID.png
