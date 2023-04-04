#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o train_diffusion_ped_loc.out

export CUDA_VISIBLE_DEVICES=1

python train_ped_loc.py
