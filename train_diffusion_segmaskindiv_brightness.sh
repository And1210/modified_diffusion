#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o train_diffusion_segmaskindiv_brightness.out

export CUDA_VISIBLE_DEVICES=4

python train_segmaskindiv_brightness.py
