#!/bin/bash
#SBATCH --time=01:00:00
#SBATCH --partition=highmemgpu
#SBATCH --gres=gpu:1

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/exports/humgen/idenhond/miniconda3/envs/enformer_dev/lib 

python collect_env_details.py