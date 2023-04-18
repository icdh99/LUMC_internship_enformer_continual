import torch

input = torch.load('/exports/humgen/idenhond/data/Enformer_train/Enformer_train_embeddings_pretrainedmodel/embeddings_seq1.pt', map_location=torch.device('cpu'))
print(f'input shape: {input.shape}')
target = torch.load('/exports/humgen/idenhond/data/Enformer_train/Enformer_train_targets/targets_seq1.pt', map_location=torch.device('cpu'))
print(f'target shape: {target.shape}')