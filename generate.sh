#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 3
#SBATCH -o generate.out

export CUDA_VISIBLE_DEVICES=4

python generate.py
