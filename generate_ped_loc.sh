#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o generate_ped_loc.out

export CUDA_VISIBLE_DEVICES=3

python generate_ped_loc.py
