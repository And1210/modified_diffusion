#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o train_diffusion_segmaskindiv_0.out

export CUDA_VISIBLE_DEVICES=1

python train_segmaskindiv_0.py
