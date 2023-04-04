#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o train_diffusion_bbox_lambda.out

export CUDA_VISIBLE_DEVICES=4

python train_bbox_lambda.py
