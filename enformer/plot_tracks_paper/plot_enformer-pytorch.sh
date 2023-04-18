#!/bin/bash
#SBATCH --mail-type=START,END,FAIL
#SBATCH --mail-user=icdenhond@gmail.com
#SBATCH --partition=highmemgpu
#SBATCH --time=01:00:00
#SBATCH --mem=50G
#SBATCH --output=Reports/output.%x.%j.out

script=/exports/humgen/idenhond/projects/enformer/plot_tracks_paper/plot_enformer-pytorch.py

echo 'Date: ' $(date)
echo 'Script: ' $script
echo 'Job name: ' $SLURM_JOB_NAME
echo 'Job ID: ' $SLURM_JOBID
echo 'Output folder: ' Reports/output.$SLURM_JOB_NAME.$SLURM_JOBID.out
echo 'Job folder: ' $SLURM_SUBMIT_DIR; echo 

python $script