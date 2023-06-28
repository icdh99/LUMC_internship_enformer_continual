#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --time=24:00:00
#SBATCH --mail-user=icdenhond@gmail.com
#SBATCH --ntasks-per-node=8
#SBATCH --output=Reports/%j.%x.out

echo 'Date: ' $(date)
echo 'Job name: ' $SLURM_JOB_NAME
echo 'Job ID: ' $SLURM_JOBID
echo 'Output folder: ' Reports/$SLURM_JOBID.$SLURM_JOB_NAME.out
echo 'Job folder: ' $SLURM_SUBMIT_DIR; echo 

# NIET VERGETEN: CONDA ACTIVATE BASENJI !!!!!!! 

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/exports/humgen/idenhond/miniconda3/envs/basenji/lib 

bed_file=/exports/humgen/idenhond/data/basenji_preprocess/unmap_macro.bed   #HUMAN
output_dir=/exports/humgen/idenhond/data/basenji_preprocess/output_tfr_human_atac # this folder should exist and contain the sequences.bed file
genome=/exports/humgen/idenhond/genomes/hg38.ml.fa
blacklist=/exports/humgen/idenhond/data/Basenji/hg38.blacklist.rep.bed
unmappable=/exports/humgen/idenhond/data/Basenji/umap_k24_t10_l32.bed #HUMAN
txt_file=/exports/humgen/idenhond/data/basenji_preprocess/targets_human_atac.txt


# ADJUST NORMALIZATION OPTION
/exports/humgen/idenhond/basenji_dev/basenji/bin/basenji_data.py -g $bed_file -b $blacklist -u $unmappable -l 131072 --local --restart --crop 8192 -o $output_dir -p 8 -w 128 --normalize false $genome $txt_file

# python inspect_tfr.py

# Next, we want to choose genomic sequences to form batches for stochastic gradient descent, divide them into training/validation/test sets, and construct TFRecords to provide to downstream programs.

# The script [basenji_data.py](https://github.com/calico/basenji/blob/master/bin/basenji_data.py) implements this procedure.

# The most relevant options here are:

# | Option/Argument | Value | Note |

# | -s | 0.1 | Down-sample the genome to 10% to speed things up here. |
# | -g | data/unmap_macro.bed | Dodge large-scale unmappable regions like assembly gaps. |
# | -l | 131072 | Sequence length. |
# | --local | True | Run locally, as opposed to on my SLURM scheduler. |
# | -o | data/heart_l131k | Output directory |
# | -p | 8 | Uses multiple concourrent processes to read/write. |
# | -t | .1 | Hold out 10% sequences for testing. |
# | -v | .1 | Hold out 10% sequences for validation. |
# | -w | 128 | Pool the nucleotide-resolution values to 128 bp bins. |
# | fasta_file| data/hg19.ml.fa | FASTA file to extract sequences from. |
# | targets_file | data/heart_wigs.txt | Target samples table with BigWig paths. |
