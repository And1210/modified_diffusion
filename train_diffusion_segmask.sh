#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o train_diffusion_segmask.out

export CUDA_VISIBLE_DEVICES=2

python train_segmask.py
