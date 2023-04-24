This folder contains the scripts necessary to train a DNN head model on the human ATAC-seq tracks from the paper 'A transcriptomic and epigenomic cell atlas of the mouse primary motor cortex'.

Raw bigwig files are derived from: https://genome.ucsc.edu/cgi-bin/hgTrackUi?hgsid=1609681239_fUFy3yi8hM5baXlnKyZFAvUuEyR5&c=chr15&g=hub_2136863_humanMopATAC%2DseqMultiwigs
-- 66 tracks

The script /exports/humgen/idenhond/projects/basenji_preprocess/make_tfr_atac_human.sh is used to create the tensorflow records (tfr).
TFRs are stored at /exports/humgen/idenhond/data/basenji_preprocess/output_tfr_human_atac/tfrecords
Target information is stored at /exports/humgen/idenhond/data/basenji_preprocess/output_tfr_human_atac/targets.txt

Example plots comparing the tensorflow records and the bw tracks in IGV are stored at /exports/humgen/idenhond/projects/basenji_preprocess/plots_human_atac

For my model training pipeline, the targets need to be stored in .pt format. 
-- /exports/humgen/idenhond/projects/enformer/dnn_head/train_human_atac/store_target_as_tensor.sh

*Model training*
Train input: /exports/humgen/idenhond/data/Enformer_train/Enformer_train_embeddings_pretrainedmodel
Validation input: /exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_embeddings_pretrainedmodel_perseq
Train targets: /exports/humgen/idenhond/data/Enformer_train/Human_ATAC_train_targets
Validation targets: /exports/humgen/idenhond/data/Enformer_validation/Human_ATAC_validation_targets



