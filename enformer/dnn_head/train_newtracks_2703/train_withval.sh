#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=icdenhond@gmail.com
#SBATCH --time=4-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=2
#SBATCH --nodes=1
#SBATCH --mem=90G
#SBATCH --output=./Reports/output.%x.%j.out

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/exports/humgen/idenhond/miniconda3/envs/enformer_dev/lib 

script=/exports/humgen/idenhond/projects/enformer/dnn_head/train_newtracks_2703/train_withval.py

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

srun python $script

# SBATCH --gpus-per-node=2