#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o train_diffusion_celeba.out

export CUDA_VISIBLE_DEVICES=3

python train_celeba.py
