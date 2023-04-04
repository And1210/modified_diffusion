#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o train_diffusion_segmaskindiv_20.out

export CUDA_VISIBLE_DEVICES=4

python train_segmaskindiv_20.py
