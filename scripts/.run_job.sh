#!/bin/bash

exec sbatch -N "$1" -n "$2" ~/game-of-life/scripts/slurm.sh "${@:3}"
