#!/bin/bash
#SBATCH --job-name=pro_DTL
#SBATCH --error=%x-%j.error
#SBATCH --out=%x-%j.out
#SBATCH --mem=80G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=1-0:0:0
#SBATCH --partition=theia,mtst,bigmem

#conda activate polyfun
python 5.Rank_Evaluate_Percentage.py $1 DTL prostate 2024 #$1 50000,100000

