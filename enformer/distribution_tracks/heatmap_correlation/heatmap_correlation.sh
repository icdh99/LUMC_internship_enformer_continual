#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=icdenhond@gmail.com
#SBATCH --time=4-10:00:00
#SBATCH --mem=5G
#SBATCH --output=Reports/%j.%x.out

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/exports/humgen/idenhond/miniconda3/envs/enformer_dev/lib 

script=/exports/humgen/idenhond/projects/enformer/distribution_tracks/heatmap_correlation/heatmap_correlation.py

echo 'Date: ' "$(date)"
echo 'Script: ' $script
echo 'Job name: ' $SLURM_JOB_NAME
echo 'Job ID: ' $SLURM_JOBID
echo 'Output folder: ' Reports/$SLURM_JOBID.$SLURM_JOB_NAME.out
echo 'Job folder: ' $SLURM_SUBMIT_DIR; echo 

# $1 = subset {test} # train_targets, train_outputs zijn te groot dus niet runnen
python $script $1