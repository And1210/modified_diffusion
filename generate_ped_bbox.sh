#!/bin/sh
#SBATCH -p Sasquatch
#SBATCH -G 1
#SBATCH -o generate_ped_bbox.out

export CUDA_VISIBLE_DEVICES=7

python generate_ped_bbox.py
