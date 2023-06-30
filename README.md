A transfer-learning approach to fine-tune existing sequence-based deep learning models to predict gene expression for new cell types 

Here, we present a transfer-learning approach to predict new genomic profiles like chromatin accessibility and transcription factor binding directly from DNA sequence. We treat Enformer, the state-of-the-art deep learning model for predicting gene expression from DNA sequence, as a pretrained model by finetuning the human head model with the Enformer embeddings. We define the downstream fine-tuning tasks as learning new genomic profiles on different resolutions, ranging from tissue to cell-type specific. This repository contains the code to finetune the human head model on the snATAC-seq tracks, from the raw genomic coverage data to evalauting the predictions.

1) Genomic coverage files
2) Make tensor flow records from genomic coverage files
3) Generate embeddings from Enformer 
4) Train human head model
5) Evaluate human head model
6) DAR analysis

1. Genomic coverage files
The snATAC genomic coverage (BigWig) files are downloaded from https://genome.ucsc.edu/cgi-bin/hgTrackUi?hgsid=1609681239_fUFy3yi8hM5baXlnKyZFAvUuEyR5&c=chr15&g=hub_2136863_humanMopATAC%2DseqMultiwigs and stored at /exports/humgen/idenhond/data/human_Mop_ATAC/bw_files. The wget or curl command can be used with /exports/humgen/idenhond/data/human_Mop_ATAC/filenames/human_Mop_ATAC.txt to download all files at once. 

2. Prepare genomic coverage files for model training
We use the basenji_data.py script from the Basenji repository to generate tensorflow records that are used to train Basenji and Enformer on. This script converts the BigWig coverage files to a tfr format.
How to run basenji_data.py is described in /exports/humgen/idenhond/projects/basenji_preprocess/README.md. 

3. Generate embeddings from Enformer 
We store the embeddings for all train, test, and validation sequences from Enformer-pytorch.
How to do this is described in /exports/humgen/idenhond/projects/enformer/get_output_emb/README.md. Here, also the output of the Enformer-pytorch model can be stored. 

Embeddings test: /exports/humgen/idenhond/data/Enformer_test/Enformer_test_embeddings
Embeddings train: /exports/humgen/idenhond/data/Enformer_train/Enformer_train_embeddings_pretrainedmodel
Embeddings validation: /exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_embeddings_pretrainedmodel_perseq

4. Train human head model
How to train the human head model on the human snATAC tracks is described in /exports/humgen/idenhond/projects/enformer/dnn_head/train_human_atac/README.md.
Targets test: /exports/humgen/idenhond/data/Enformer_test/Human_ATAC_test_targets
Targets train: /exports/humgen/idenhond/data/Enformer_train/Human_ATAC_train_targets
Targets validation: /exports/humgen/idenhond/data/Enformer_validation/Human_ATAC_validation_targets

Human head model trained on the 66 snATAC tracks: /exports/humgen/idenhond/projects/enformer/dnn_head/train_human_atac/model_2023-04-24 17:53:40.485828/epoch=18-step=5054-val_loss=0.4.ckpt

5. Evaluate human head model
Evaluation of the performance of the human head model trained on the snATAC tracks is described in /exports/humgen/idenhond/projects/enformer/correlation/README.md.

6. DAR (differentially accessible regions analysis)
This is further described in /exports/humgen/idenhond/projects/dar/README.md

Other important scripts
Boxplots as in figure 1D, 2B, and 3C are generated with /exports/humgen/idenhond/projects/enformer/distribution_tracks/boxplot_correlation/boxplot.py and /exports/humgen/idenhond/projects/enformer/distribution_tracks/boxplot_correlation/boxplot_atac.py.
Predicted and observed tracks are made with the scripts in /exports/humgen/idenhond/projects/enformer/plot_tracks_paper/Paper_figures
Analysis and visualization of the weights learned by Enformer-pytorch and the human head model is done in /exports/humgen/idenhond/projects/enformer/weigths/plot_weights_figure1.py

Human ATAC clusters
As I trained the human head model for snATAC data on 66 clusters and removed some of the clusters in the analysis, here is some guidance on how to analyse the predictions of the snATAC model.

The tensors predicted by the model have the shape of torch.Size([1, 896, 66]) (e.g. /exports/humgen/idenhond/data/Enformer_test/Enformer_test_output_humanatac/output_seq1.pt)
This means that for the first sequence, the model predicted 66 tracks with 896 bins for each track. 
The index column in /exports/humgen/idenhond/data/basenji_preprocess/targets_human_atac.txt indicates the name for each track in correct order.

I created a second targets file for each level (Class, Subclass, AC-level) that can be used to subset the correct tracks per level. 
/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_Ac-level_cluster.csv = 20 subclasses
/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_Subclass.csv = 48 ac level clusters
/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_Class.csv = 3 classes

