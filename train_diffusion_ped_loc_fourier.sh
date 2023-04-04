#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o train_diffusion_ped_loc_fourier.out

export CUDA_VISIBLE_DEVICES=2

python train_ped_loc_fourier.py
