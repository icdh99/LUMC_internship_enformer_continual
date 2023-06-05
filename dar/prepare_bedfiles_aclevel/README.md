This folder contains the scripts to run the DAR analysis on supplementary table 14A from the Bakken paper

Goal: show the predicted ATAC tracks for the DAR sequences that overlap with the Enformer test regions on the AC cluster level

step 1
- prepare Enformer test bed file (/exports/humgen/idenhond/projects/dar/prepare_enformer_test.sh)

step 2 
- analyse sequences from table 14A (prepare_bedfile14A.py)
- create bedfile from table 14A (prepare_bedfile14A.py)
- make plots

step 3 
- perform intersect between 14A and Enformer test (intersect.sh)

step 4 
- analyse intersected sequences (analyse_intersect.py)
- make plots 

step 5 
- get ATAC model output for intersect sequences
- plot (store_outptut_intersect.py)