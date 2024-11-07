#!/bin/bash
#SBATCH --job-name=TWAS_bre
#SBATCH --error=%x-%j.error
#SBATCH --out=%x-%j.out
#SBATCH --mem=10G
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=7-0:0:0
#SBATCH --partition=cpu2021,cpu2022,cpu2023,cpu2019,theia

Rscript TWAS_modelbuilding_GTEx.R $1 $2
