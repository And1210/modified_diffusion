#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o generate_bbox_1.out

export CUDA_VISIBLE_DEVICES=4

python generate_bbox_1.py
