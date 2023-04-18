#!/bin/bash
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=icdenhond@gmail.com
#SBATCH --time=01:00:00
#SBATCH --mem=60G
#SBATCH --output=Reports/%j.%x.out

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/exports/humgen/idenhond/miniconda3/envs/enformer_dev/lib 

script=/exports/humgen/idenhond/projects/enformer/distribution_tracks/distribution.py

echo 'Date: ' $(date)
echo 'Script: ' $script
echo 'Job name: ' $SLURM_JOB_NAME
echo 'Job ID: ' $SLURM_JOBID
echo 'Output folder: ' Reports/$SLURM_JOBID.$SLURM_JOB_NAME.out
echo 'Job folder: ' $SLURM_SUBMIT_DIR; echo 

python $script