import torch
from torch.utils.data import DataLoader, Dataset
from avalanche.benchmarks.utils import AvalancheDataset

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
        y = torch.load(f'/exports/humgen/idenhond/data/Enformer_train/Enformer_train_targets/targets_seq{ID}.pt', map_location=torch.device('cpu'))
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
        y = torch.load(f'/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_targets_perseq/targets_seq{ID}.pt', map_location=torch.device('cpu'))
        # x = torch.rand(100, 22)
        # y = torch.rand(100, 22)
        y = torch.squeeze(y)
        return x, y
    
BATCH_SIZE = 64
EPOCHS = 20
NUM_WORKERS = 1
    
partition_indices = {}
partition_indices['train'] = list(range(1, 34021+1))
partition_indices['val'] = list(range(1, 2213+1))

training_set = MyDataset_train(partition_indices['train'])
print(f'number of train samples: {len(training_set)}')
training_generator = DataLoader(training_set, 
                                batch_size = BATCH_SIZE, 
                                shuffle = True, # shuffle sample volgorde in elke epoch? zoiets
                                num_workers = NUM_WORKERS,
                                pin_memory = True)
print(f'number of train batches: {len(training_generator)}')

val_set = MyDataset_val(partition_indices['val'])

val_data_avl = AvalancheDataset(val_set)
print(f'number of val samples: {len(val_set)}')
val_generator = DataLoader(val_data_avl, 
                            batch_size = BATCH_SIZE, 
                            shuffle = False, # shuffle sample volgorde in elke epoch? zoiets
                            num_workers = NUM_WORKERS,
                            pin_memory = True)
print(f'number of validation batches: {len(val_generator)}')


# for i, (x, y) in enumerate(val_generator):
#     pass
# print(f'num mini-batch processed: {i + 1}')
