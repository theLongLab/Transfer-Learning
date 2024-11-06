#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2   # maximum CPU cores per GPU request: 6 on Cedar, 16 on Graham.
#SBATCH --mem=10G        # memory per node
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=7-00:00:00
#SBATCH --account=def-qingrunz-ab
#SBATCH --job-name=pro
#SBATCH --error=%N-%j.error
#SBATCH --output=%N-%j.out

###Cedar
#module load StdEnv/2020  #cudnn/8.2.0, cuda/11.4
#module load python/3.8.2

#source /home/qingli2/scratch/tensorflow2_4_1/bin/activate
#pip install tensorflow==2.4.1 --no-deps
#pip install dm-sonnet==2.0.0
#pip install kipoiseq 
#pip install numpy==1.20.0 
#pip instals pandas==1.2.3
#pip installs protobuf==3.9.1

python 2.enformer-usage_SNPs_q_CC_DTL.py $1 

