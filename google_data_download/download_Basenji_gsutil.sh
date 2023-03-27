#!/bin/bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=icdenhond@gmail.com
#SBATCH --time=2-12:00:00

# gsutil -m -u dataenformer cp -r gs://basenji_barnyard/data/human/tfrecords /exports/humgen/idenhond/Basenji_data

# gsutil -u dataenformer cp gs://basenji_barnyard/hg38.ml.fa.gz /exports/humgen/idenhond/Basenji_data
# # gunzip hg38.ml.fa.gz
# gsutil -u dataenformer cp gs://basenji_barnyard/mm10.ml.fa.gz /exports/humgen/idenhond/Basenji_data && gunzip mm10.ml.fa.gz
# gsutil -u dataenformer cp gs://basenji_barnyard/data/mouse/sequences.bed /exports/humgen/idenhond/Basenji_data/mouse-sequences.bed
# gsutil -u dataenformer cp gs://basenji_barnyard/data/human/statistics.json /exports/humgen/idenhond/Basenji_data/human-statistics.json
# gsutil -u dataenformer cp gs://basenji_barnyard/data/human/targets.txt /exports/humgen/idenhond/Basenji_data/human-targets.txt
# gsutil -u dataenformer cp gs://basenji_barnyard/targets_human.txt /exports/humgen/idenhond/Basenji_data
# gsutil -u dataenformer cp gs://basenji_barnyard/targets_mouse.txt /exports/humgen/idenhond/Basenji_data


gsutil -u dataenformer cp gs://basenji_barnyard/hg38.blacklist.rep.bed
gsutil -u dataenformer cp gs://basenji_barnyard/hg38_gaps.bed
gsutil -u dataenformer cp gs://basenji_barnyard/umap_k24_t10_l32.bed