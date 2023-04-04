#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 3
#SBATCH -o generate_mnist.out

export CUDA_VISIBLE_DEVICES=1

python generate_mnist.py
