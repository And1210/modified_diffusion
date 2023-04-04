#!/bin/sh
#SBATCH -p Combined
#SBATCH --nodelist aurora
#SBATCH -o train_diffusion_bbox_MOT17.out

export CUDA_VISIBLE_DEVICES=6

python train_bbox_MOT17.py
