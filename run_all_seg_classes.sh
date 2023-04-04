for i in 0 1 2 4 5 7 8 9 10 11 12 13 14
do
	sbatch train_diffusion_segmaskindiv_$i.sh
done
