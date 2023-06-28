This repository contains code to preprocess epigenomic tracks into files for the enformer/basenji model

preprocess.ipynb --> komt van /exports/humgen/idenhond/basenji_dev/basenji/tutorials/preprocess.ipynb

Requirement: install basenji (https://github.com/calico/basenji)

Make_tfr scripts: make tensor flow record files for a specific target.txt file that indicates the bigwig files to be converted to a .tfr format. 
inspect_tfr scripts: scripts that plot targets/outputs (old)

HOW TO RUN (example for human snATAC tracks)
Working folder: /exports/humgen/idenhond/data/basenji_preprocess/output_tfr_human_atac
Make the file /exports/humgen/idenhond/data/basenji_preprocess/targets_human_atac.txt with the script /exports/humgen/idenhond/projects/basenji_preprocess/make_targets_file_humanAtac.sh.
The targets file indicates which BigWig files should be converted to .tfr format and sets some parameters. I used clip 32, scale 2, and sum_stat mean similar to the ATAC tracks in Enformer.
This file has a structure similar to Supplementary table 2 and 3 from the Enformer paper. 

Make the tfr files: /exports/humgen/idenhond/projects/basenji_preprocess/make_tfr_atac_human.sh
To run this script, the desired output folder (in this case /exports/humgen/idenhond/data/basenji_preprocess/output_tfr_human_atac) should exist and contain only a sequences.bed file. 
The sequences.bed file indicates the train, test and validation sequences. I have used the Enfomer sequences.bed file (/exports/humgen/idenhond/data/Basenji/sequences.bed) but this can be any bed file.
Other files needed to run this script: 
- bed_file, blacklist, unmapple = skipped regions, from Enformer/Basenji
- genome = hg38 genome
- txt_file = targets file

Output after running /exports/humgen/idenhond/projects/basenji_preprocess/make_tfr_atac_human.sh:
/exports/humgen/idenhond/data/basenji_preprocess/output_tfr_human_atac contains the following:
- seqs_cov: intermediate files for basenji_data.py
- tfrecords: final tfrecords for train, test and validation. each tf record contains 256 sequences. 
- sequences.bed (same as above)
- statistics.json: information about the tf records
- targets.txt: same as above



