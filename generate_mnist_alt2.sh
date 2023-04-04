#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 3
#SBATCH -o generate_mnist_alt2.out

export CUDA_VISIBLE_DEVICES=7

python generate_mnist_alt2.py
