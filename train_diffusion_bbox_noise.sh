#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o train_diffusion_bbox_noise.out

export CUDA_VISIBLE_DEVICES=7

python train_bbox_noise.py
