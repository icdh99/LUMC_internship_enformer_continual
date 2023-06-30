import torch
from torch.utils.data import Dataset, DataLoader
import random

"""
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel

partition: {'train': ['id-1', 'id-2', 'id-3'], 'validation': ['id-4']}
list of training/validation IDs --> zelf maken

"""
class MyDataset_train(Dataset):
    def __init__(self, list_IDs):
        assert type(list_IDs) == list, f'list IDs expected type list but got {type(list_IDs)}'
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        ID = self.list_IDs[index]
        # load data and get label
        x = torch.load(f'/exports/humgen/idenhond/data/Enformer_train/Enformer_train_embeddings_pretrainedmodel/embeddings_seq{ID}.pt', map_location=torch.device('cpu')) 
        y = torch.load(f'/exports/humgen/idenhond/data/Enformer_train/Human_ATAC_train_targets/targets_seq{ID}.pt', map_location=torch.device('cpu')) 
        y = torch.squeeze(y)
        return x, y

class MyDataset_val(Dataset):
    def __init__(self, list_IDs):
        assert type(list_IDs) == list, f'list IDs expected type list but got {type(list_IDs)}'
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        ID = self.list_IDs[index]
        # load data and get label
        x = torch.load(f'/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_embeddings_pretrainedmodel_perseq/embeddings_seq{ID}.pt', map_location=torch.device('cpu')) 
        y = torch.load(f'/exports/humgen/idenhond/data/Enformer_validation/Human_ATAC_validation_targets/targets_seq{ID}.pt', map_location=torch.device('cpu')) 
        y = torch.squeeze(y)
        return x, y
