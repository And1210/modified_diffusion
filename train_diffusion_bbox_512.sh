#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o train_diffusion_bbox_512.out

export CUDA_VISIBLE_DEVICES=0,3,7

python train_bbox_512.py
