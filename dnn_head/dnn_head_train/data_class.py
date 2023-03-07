import torch
from torch.utils.data import Dataset, DataLoader
import random

"""
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

partition: {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
list of training/validation IDs --> zelf maken

"""
class MyDataset(Dataset):
    def __init__(self, list_IDs):
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        ID = self.list_IDs[index]
        # load data and get label
        # x = torch.load(f'/exports/humgen/idenhond/data/Enformer_train/Enformer_train_embeddings_pretrainedmodel/embeddings_seq{ID}.pt')
        y = torch.load(f'/exports/humgen/idenhond/data/Enformer_train/Enformer_train_targets/targets_seq{ID}.pt')
        return y
    
