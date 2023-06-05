import torch
import numpy as np

emb_big = torch.load('/exports/archive/hg-funcgenom-research/idenhond/Enformer_test/Enformer_test_embeddings_newmodel/embeddings_test_pretrainedmodel.pt')
print(emb_big.shape) # torch.Size([1937, 896, 3072])

for i, tensor in enumerate(emb_big):
    print(i)
    print(tensor.shape) # torch.Size([896, 3072])
    tensor = tensor.numpy()
    np.savetxt(f'/exports/humgen/idenhond/data/Enformer_test/Enformer_test_embeddings/embeddings_seq{i+1}.txt', tensor, delimiter=",")
    # torch.save(tensor, f'/exports/humgen/idenhond/data/Enformer_test/Enformer_test_embeddings/embeddings_seq{i+1}.pt')

# device = 'cpu'
# emb = torch.load(f'/exports/humgen/idenhond/data/Enformer_train/Enformer_train_embeddings_pretrainedmodel/embeddings_seq{1}.pt', map_location=torch.device(device))
# print(emb.shape) # torch.Size([896, 3072])