#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o train_diffusion_bbox_C16gen.out

export CUDA_VISIBLE_DEVICES=7

python train_bbox_C16gen.py
