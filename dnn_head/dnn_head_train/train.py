from datetime import datetime
start = datetime.now()
import torch
import torch.nn as nn
import pytorch_lightning as pl
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import numpy as np
from data_class import MyDataset
from model_class import model
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from datetime import datetime
import os
import random

# # setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device (set to GPU if available):', device)

### make partitiion dictionary with IDs for train and validation
# set random seed
random.seed(18)

all_indices = set(range(1, 34021+1)) #start inclusive, end exclusive
print(f'number of indices: {len(list(all_indices))}')
print(f'first index: {list(all_indices)[0]}')
print(f'last index: {list(all_indices)[-1]}')

fraction_train = 0.8
fraction_val = 1 - fraction_train

nr_train = len(list(all_indices)) * fraction_train
nr_val = len(list(all_indices)) * fraction_val
print(f'number of indices for train: {int(nr_train)}')
print(f'number of indices for val: {int(nr_val)}')
assert nr_train + nr_val == len(list(all_indices)), f"expected 34021 samples for train + val, got: {nr_train + nr_val}"

indices_train = set(random.sample(all_indices, int(nr_train)))
indices_val = all_indices - indices_train

print(f'number of indices for train: {len(list(indices_train))}')
print(f'number of indices for val: {len(list(indices_val))}')

partition_indices = {}
partition_indices['train'] = indices_train
partition_indices['val'] = indices_val

# generators
training_set = MyDataset(partition_indices['train'])
print(f'number of train samples: {len(training_set)}')

training_generator = DataLoader(training_set, 
                                batch_size = 32, 
                                shuffle = True, # shuffle sample volgorde in elke epoch? zoiets
                                num_workers = 6)
print(f'number of batches in training generator: {len(training_generator)}')

## make folder to store model in
ts = datetime.timestamp(datetime.now())
date_time = datetime.fromtimestamp(ts)
if not os.path.exists(f'./model_{date_time}/'):
	os.makedirs(f'./model_{date_time}/')

# define callbacks 
early_stop_callback = EarlyStopping(monitor="val_loss", 
                                            min_delta=0.00, 
                                            patience=5, verbose=False, mode="min")

modelcheckpoint = ModelCheckpoint(monitor = 'val_loss', 
								mode = 'min', 
								save_top_k = 1, 
								dirpath = f'./model_{date_time}/', 
								filename = "{epoch}-{step}-{val_loss:.1f}")

callbacks = [RichProgressBar(), early_stop_callback, modelcheckpoint]

BATCH_SIZE = 32
EPOCHS = 20

clf = model()
trainer = pl.Trainer(max_epochs = EPOCHS, callbacks = callbacks, enable_checkpointing = True) # , accelerator = 'gpu', devices = 1  # TODO: naar gpu zetten!! 
# trainer.fit(clf, trainloader, valloader)