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



