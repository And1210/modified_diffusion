#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o train_diffusion_ped_tmp.out

export CUDA_VISIBLE_DEVICES=7

python train_ped_tmp.py
