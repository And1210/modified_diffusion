#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o train_diffusion_segmaskindiv_13.out

export CUDA_VISIBLE_DEVICES=7

python train_segmaskindiv_13.py
