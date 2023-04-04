#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o train_diffusion_segmaskindiv_18.out

export CUDA_VISIBLE_DEVICES=2

python train_segmaskindiv_18.py
