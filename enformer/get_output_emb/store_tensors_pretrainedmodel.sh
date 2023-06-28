#!/bin/bash
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=icdenhond@gmail.com
#SBATCH --time=12:00:00
#SBATCH --partition=highmemgpu
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --output=./Reports/output.%x.%j.out

script=/exports/humgen/idenhond/projects/enformer/get_output_emb/store_tensors_pretrainedmodel.py

echo 'Date: ' $(date)
echo 'Script: ' $script
echo 'Job name: ' $SLURM_JOB_NAME
echo 'Job ID: ' $SLURM_JOBID
echo 'Output folder: ' ./Reports/output.$SLURM_JOB_NAME.$SLURM_JOBID.out
echo 'Job folder: ' $SLURM_SUBMIT_DIR; echo 

# echo 'Number of CPUs on the allocated node: ' $SLURM_CPUS_ON_NODE
# echo 'Number of CPUs requested per task: ' $SLURM_CPUS_PER_TASK
# echo 'Number of GPUs requested: ' $SLURM_GPUS
# echo 'Requested GPU count per allocated node: ' $SLURM_GPUS_PER_NODE
# echo 'Requested GPU count per allocated task: ' $SLURM_GPUS_PER_TASK

# arg 1 = subset
subset=$1
echo 'Subset: ' $subset # test, train or validss
python $script $subset

