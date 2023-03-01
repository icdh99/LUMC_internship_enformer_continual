#!/bin/bash
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=icdenhond@gmail.com
#SBATCH --time=1-00:00:00
#SBATCH --mem=150G
#SBATCH --output=Reports/output.%x.%j.out

script=/exports/humgen/idenhond/projects/enformer/get_output_emb/store_tensors_onefile.py

echo 'Date: ' $(date)
echo 'Script: ' $script
echo 'Job name: ' $SLURM_JOB_NAME
echo 'Job ID: ' $SLURM_JOBID
echo 'Output folder: ' Reports/output.$SLURM_JOB_NAME.$SLURM_JOBID.out
echo 'Job folder: ' $SLURM_SUBMIT_DIR; echo 

# arg 1 = subset
subset=$1
echo 'Subset: ' $subset
python $script $subset