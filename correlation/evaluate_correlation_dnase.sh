#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=icdenhond@gmail.com
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --output=Reports/%j.%x.out

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/exports/humgen/idenhond/miniconda3/envs/enformer_dev/lib 

script=/exports/humgen/idenhond/projects/enformer/correlation/evaluate_correlation_dnase.py

echo 'Date: ' $(date)
echo 'Script: ' $script
echo 'Job name: ' $SLURM_JOB_NAME
echo 'Job ID: ' $SLURM_JOBID
echo 'Output folder: ' Reports/$SLURM_JOBID.$SLURM_JOB_NAME.out
echo 'Job folder: ' $SLURM_SUBMIT_DIR; echo 

# arg 1 = subset
# arg 2 = nr steps
subset=$1
nr_steps=$2
echo 'Subset: ' $subset
echo 'Number of steps: ' $nr_steps
python $script $subset $nr_steps