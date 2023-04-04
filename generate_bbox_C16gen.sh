#!/bin/sh
#SBATCH -p Combined
#SBATCH --nodelist aurora
#SBATCH -o generate_bbox_C16gen.out

export CUDA_VISIBLE_DEVICES=3

python generate_bbox_C16gen.py
