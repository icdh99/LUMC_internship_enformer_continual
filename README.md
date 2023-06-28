A transfer-learning approach to fine-tune existing sequence-based deep learning models to predict gene expression for new cell types 

Here, we present a transfer-learning approach to predict new genomic profiles like chromatin accessibility and transcription factor binding directly from DNA sequence. We treat Enformer, the state-of-the-art deep learning model for predicting gene expression from DNA sequence, as a pretrained model by finetuning the human head model with the Enformer embeddings. We define the downstream fine-tuning tasks as learning new genomic profiles on different resolutions, ranging from tissue to cell-type specific. This repository contains the code to finetune the human head model on the snATAC-seq tracks, from the raw genomic coverage data to evalauting the predictions.


1) Genomic coverage files
2) Make tensor flow records from genomic coverage files
3) Generate embeddings from Enformer 
4) Train human head model
5) Evaluate human head model

1. Genomic coverage files
The snATAC genomic coverage (BigWig) files are downloaded from https://genome.ucsc.edu/cgi-bin/hgTrackUi?hgsid=1609681239_fUFy3yi8hM5baXlnKyZFAvUuEyR5&c=chr15&g=hub_2136863_humanMopATAC%2DseqMultiwigs and stored at /exports/humgen/idenhond/data/human_Mop_ATAC/bw_files. 

2. Prepare genomic coverage files for model training
We use the basenji_data.py script from the Basenji repository to generate tensorflow records that are used to train Basenji and Enformer on. This script converts the BigWig coverage files to a tfr format. Next, we convert the tfr files to torch files to be used for model input. (not optimal) 

How to run:

3. Generate embeddings from Enformer 
4. Train human head model
5. Evaluate human head model




Plots in report
1.
2.
3.
4.
5.
S1
S2
S3
S4
S5
