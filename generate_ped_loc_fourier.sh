#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o generate_ped_loc_fourier.out

export CUDA_VISIBLE_DEVICES=4

python generate_ped_loc_fourier.py
