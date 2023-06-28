This folder contains the scripts to generate the output and embedding tensors from the Enformer model 

# CHECK PATHS IN SCRIPTS BEFORE RUNNING

- stored in .pt format

scripts: (generates one .pt file per sequence)
/exports/humgen/idenhond/projects/enformer/get_output_emb/store_tensors_test.py
/exports/humgen/idenhond/projects/enformer/get_output_emb/store_tensors_validation.py
data: 
/exports/humgen/idenhond/data/Enformer_test/Enformer_test_embeddings
/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output
/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_embeddings
/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_output

bed file used for input (human_sequences.bed)

22/2 NEW MODEL
script:
/exports/humgen/idenhond/projects/enformer/get_output_emb/store_tensors_othermodel.py
data:
/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_output_newmodel
/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_output_newmodel

6/3: store embedding from train sequences in a .pt file per sequence
script: projects/enformer/get_output_emb/store_tensors_pretrainedmodel.py
data: data/Enformer_train/Enformer_train_embeddings_pretrainedmodel

HOW TO STORE THE EMBEDDINGS AND OUTPUT FROM ENFORMER-PYTORCH
This folder contains the code to store the output and embeddings from Enformer-pytorch for a defined set of input sequences.
/exports/humgen/idenhond/projects/enformer/get_output_emb/store_tensors_pretrainedmodel.sh with option test, train or valid
- in the python script, you can choose at what locationss to store the output and the embedding of each sequence 

Outputs
- /exports/humgen/idenhond/data/Enformer_test/Enformer_test_output
- /exports/archive/hg-funcgenom-research/idenhond/Enformer_validation/Enformer_validation_output_newmodel

Embeddings
- /exports/humgen/idenhond/data/Enformer_test/Enformer_test_embeddings
- /exports/humgen/idenhond/data/Enformer_train/Enformer_train_embeddings_pretrainedmodel
- /exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_embeddings_pretrainedmodel_perseq