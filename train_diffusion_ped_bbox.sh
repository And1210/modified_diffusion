#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o train_diffusion_ped_bbox.out

export CUDA_VISIBLE_DEVICES=4,5,6,7

python train_ped_bbox.py
