#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=icdenhond@gmail.com
#SBATCH --time=03:00:00
#SBATCH --mem=400G
#SBATCH --output=./Reports/output.%x.%j.out
#SBATCH --array=101-132

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/exports/humgen/idenhond/miniconda3/envs/enformer_dev/lib 

script=/exports/humgen/idenhond/projects/enformer/dnn_head/tfr_as_hdf5.py

echo 'Date: ' $(date)
echo 'Script: ' $script
echo 'Job name: ' "$SLURM_JOB_NAME"
echo 'Job ID: ' "$SLURM_JOBID"
echo 'Output folder: ' ./Reports/output."$SLURM_JOB_NAME".$SLURM_JOBID.out
echo 'Job folder: ' $SLURM_SUBMIT_DIR; echo 

# echo 'Number of CPUs on the allocated node: ' $SLURM_CPUS_ON_NODE
# echo 'Number of CPUs requested per task: ' $SLURM_CPUS_PER_TASK
# echo 'Number of GPUs requested: ' $SLURM_GPUS
# echo 'Requested GPU count per allocated node: ' $SLURM_GPUS_PER_NODE
# echo 'Requested GPU count per allocated task: ' $SLURM_GPUS_PER_TASK

FILE=/exports/humgen/idenhond/data/Basenji/tfrecords/train-0-$SLURM_ARRAY_TASK_ID.tfr
# FILE="/exports/humgen/idenhond/data/Basenji/tfrecords/train-0-$SLURM_ARRAY_TASK_ID.tfr"

echo "Processing $FILE file..."
python $script $FILE

