#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o train_diffusion_bbox.out

export CUDA_VISIBLE_DEVICES=5

python train_bbox.py
