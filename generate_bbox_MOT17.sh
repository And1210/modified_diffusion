#!/bin/sh
#SBATCH -p Combined
#SBATCH --nodelist aurora
#SBATCH -o generate_bbox_MOT17.out

export CUDA_VISIBLE_DEVICES=0

python generate_bbox_MOT17.py
