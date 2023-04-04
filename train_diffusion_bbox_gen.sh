#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o train_diffusion_bbox_gen.out

export CUDA_VISIBLE_DEVICES=6

python train_bbox_gen.py
