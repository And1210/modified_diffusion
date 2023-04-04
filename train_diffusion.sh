#!/bin/sh
#SBATCH -p Combined
#SBATCH --nodelist aurora
#SBATCH -o train_diffusion.out

#export CUDA_VISIBLE_DEVICES=1

CUDA_LAUNCH_BLOCKING=1 python train.py
