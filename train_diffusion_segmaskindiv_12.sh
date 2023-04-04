#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o train_diffusion_segmaskindiv_12.out

export CUDA_VISIBLE_DEVICES=6

python train_segmaskindiv_12.py
