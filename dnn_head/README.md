3/3/2023

This folder contains the code to prepare the data to train a DNN on the embeddings of the train sequences from enformer

Data:
data/Enformer_train/Enformer_train_targets

Scripts:
store_target_as_tensor.py --> store each target from the tfr records in a seperate .pt file

Redundant scripts:
merge_embeddingstensors --> make four files into one (400G), never done
dnn_train.py --> opzet voor trainen model met train embeddings, maar hier moet alle data in het geheugen geladen worden. dus maak embedding & target per sequence in los bestand. uiteindelijke code voor trainen in /dnn_head_train

NB: the embeddings are stored in seperate .pt files with the script /exports/humgen/idenhond/projects/enformer/get_output_emb/store_tensors_pretrainedmodel.py
data: /exports/humgen/idenhond/data/Enformer_train/Enformer_train_embeddings_pretrainedmodel

