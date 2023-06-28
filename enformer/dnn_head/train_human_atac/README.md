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

HOW TO TRAIN THE HUMAN HEAD MODEL ON THE snATAC TRACKS
1 - convert the tfr files to .pt files that can be used as model input
/exports/humgen/idenhond/projects/enformer/dnn_head/train_human_atac/store_target_as_tensor.sh with option train, test or validation
Targets: 
/exports/humgen/idenhond/data/Enformer_test/Human_ATAC_test_targets
/exports/humgen/idenhond/data/Enformer_train/Human_ATAC_train_targets
/exports/humgen/idenhond/data/Enformer_validation/Human_ATAC_validation_targets

2 - run train_with_val.sh
this script generates a train and validation dataset, sets up the evaluation metrics and log folders, and trains the model.
you can adjust the name of the model in the python script. 
model_class.py: contains the model architecture. Make sure that the out_features parameter matches the number of tracks.
data_class_withval.py: loads the input (embeddings) and target for each train and validation sample (sequence)  as requested by the train and validation dataloader in the train script. 

Each run of train_with_val.sh generates two folders with the name "model" and then the date, e.g. /exports/humgen/idenhond/projects/enformer/dnn_head/train_human_atac/model_2023-04-24 17:53:40.485828. One folder contains the trained model, the other folder is always empty (there are two folders because we use two GPU's). In the reports folder, the model training results can be found under the correct slurm output file.



