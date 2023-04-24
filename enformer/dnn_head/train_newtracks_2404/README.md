This folder contains the scripts necessary to train a DNN head model on 27 new tracks


The script /exports/humgen/idenhond/projects/basenji_preprocess/make_tfr.sh is used to create the tensorflow records (tfr).
TFRs are stored at /exports/humgen/idenhond/data/basenji_preprocess/output_tfr_27newtracks_human/tfrecords
Target information is stored at /exports/humgen/idenhond/data/basenji_preprocess/output_tfr_27newtracks_human/targets.txt

Example plots comparing the tensorflow records and the bw tracks in IGV are stored at /exports/humgen/idenhond/projects/basenji_preprocess/plots

For my model training pipeline, the targets need to be stored in .pt format. 
-- /exports/humgen/idenhond/projects/enformer/dnn_head/train_newtracks_2404/store_target_as_tensor.sh
.pt files are stored at /exports/humgen/idenhond/data/Enformer_validation/Newtracks_2404_validation_targets and /exports/humgen/idenhond/data/Enformer_train/Newtracks_2404_train_targets



