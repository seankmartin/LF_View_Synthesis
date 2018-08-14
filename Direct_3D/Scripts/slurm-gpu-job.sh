#!/bin/bash
#SBATCH -p gpu-nodes
#SBATCH --gres=gpu:1

srun --gres=gpu:1 vrun.sh "$@"
wait
