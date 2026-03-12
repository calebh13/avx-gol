#!/bin/bash
#SBATCH --job-name=mpi_job
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --time=00:05:00
#SBATCH --export=ALL

# echo "Environment on $(hostname)"
srun "$@"
