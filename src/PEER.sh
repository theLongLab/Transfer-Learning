#!/bin/bash
#SBATCH --job-name=peer
#SBATCH --error=%x-%j.error
#SBATCH --out=%x-%j.out
#SBATCH --mem=40G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7-0:0:0
#SBATCH --partition=cpu2021,cpu2022,cpu2019,theia,mtst

#conda activate peer
Rscript bulk_GTEX_PEER.R $1 #prostate

