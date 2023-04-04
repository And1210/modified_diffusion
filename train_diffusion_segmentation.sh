#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o train_diffusion_segmentation.out

export CUDA_VISIBLE_DEVICES=4

CUDA_LAUNCH_BLOCKING=1 python train_segmentation.py
