#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o generate_bbox_10.out

export CUDA_VISIBLE_DEVICES=2

python generate_bbox_10.py
