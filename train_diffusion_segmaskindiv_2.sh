#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o train_diffusion_segmaskindiv_2.out

export CUDA_VISIBLE_DEVICES=3

python train_segmaskindiv_2.py
