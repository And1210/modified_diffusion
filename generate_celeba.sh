#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o generate_celeba.out

export CUDA_VISIBLE_DEVICES=4

python generate_celeba.py
