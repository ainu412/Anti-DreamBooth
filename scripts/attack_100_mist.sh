#!/bin/sh
#SBATCH --job-name=gpujob
#SBATCH --gpus=a100-40:1

#SBATCH --job-name=mist
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ziyi.guo@nus.edu.sg
#SBATCH --partition=gpu-long
#SBATCH --time=15:00:00

python attacks/mist.py --dataset_name celeb25 --img_ids ariana beyonce bruce cristiano ellen emma george jackie james johnny justin kate leonardo lucy morgan oprah rihanna shah shirley taylor copy copy2 copy3 copy4 copy5

