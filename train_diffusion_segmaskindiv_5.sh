#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o train_diffusion_segmaskindiv_5.out

export CUDA_VISIBLE_DEVICES=6

python train_segmaskindiv_5.py
