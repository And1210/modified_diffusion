#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o train_diffusion_mnist_alt2.out

export CUDA_VISIBLE_DEVICES=0

python train_mnist_alt2.py
