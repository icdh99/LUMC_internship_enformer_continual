from datetime import datetime
start = datetime.now()
import torch
import torch.nn as nn
import pytorch_lightning as pl
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import RichProgressBar
from natsort import natsorted
import glob
import tensorflow as tf

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device (set to GPU if available):', device)

### load training data
# input (embeddings)
def get_input(path):
    t_input = torch.load(path, map_location=torch.device(device))
    print(f'first tensor of train embeddings input X')
    print(f'shape: {t_input.shape}')
    print(f'device: {t_input.device}')
    print(f'number of train sequences: {len(t_input)}')
    return t_input

# target (targets)
def make_parser(): #, rna_mode
    def parse_proto(example_protos):
        """Parse TFRecord protobuf."""
        feature_spec = {
        'sequence': tf.io.FixedLenFeature([], dtype = tf.string),
        'target': tf.io.FixedLenFeature([], dtype = tf.string),
        }
        feature_tensors = tf.io.parse_single_example(example_protos, features=feature_spec)
        target = tf.io.decode_raw(feature_tensors['target'], tf.float16)
        target = tf.reshape(target, (896, 5313))
        target = tf.cast(target, tf.float32)
        return target
    return parse_proto

def file_to_records(filename):
    return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

def get_target(subset = 'train'):
    tfr_path = f'/exports/humgen/idenhond/data/Basenji/tfrecords/{subset}*.tfr'
    tfr_files = natsorted(glob.glob(tfr_path))
    print(f'number of tfr files for {subset} subset: {len(tfr_files)}')
    dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
    dataset = dataset.flat_map(file_to_records)
    dataset = dataset.map(make_parser())
    dataset = dataset.batch(1)

    targets = []
    for targets1 in dataset:
        targets.append(targets1.numpy())
    
    targets = np.array(targets)
    targets = np.squeeze(targets)
    print(f'shape targets: {targets.shape}')
    print(f'type targets: {type(targets)}')
    print(f'number of targets: {len(targets)}')
    return targets

path = '/exports/humgen/idenhond/data/Enformer_train/embeddings_train_pretrainedmodel_one.pt'
t_input = get_input(path) # tensor , 8503 entries
targets = get_target() # numpy array, 34021 entries

targets = targets[:8503, :, : ]
print(f'number of input sequences: {t_input.shape}')
print(f'number of targets: {targets.shape}')



### train val split
## train test split for X and Y
X_train, X_test, Y_train, Y_test = train_test_split(t_input.numpy(), targets, test_size = 0.20, random_state = 42)
print(f'shape of X train: {X_train.shape}')
print(f'shape of Y train: {Y_train.shape}')
print(f'shape of X test: {X_test.shape}')
print(f'shape of Y test: {Y_test.shape}')


### class Data

### class model

### define model parameters

### DataLoaders

### callbacks

### train model

### save model

