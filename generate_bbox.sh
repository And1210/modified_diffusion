#!/bin/sh
#SBATCH -p Combined
#SBATCH --nodelist aurora
#SBATCH -o generate_bbox.out

export CUDA_VISIBLE_DEVICES=5

python generate_bbox.py
