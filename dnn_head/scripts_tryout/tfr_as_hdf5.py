from datetime import datetime
start = datetime.now()
import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
from natsort import natsorted
import glob
import tensorflow as tf
import h5py
import os
import sys

## save targets from tensor flow records and embeddings as np arrays in hdf5 files
# per tfr file: save targets as hdf 5 file
# sanity check: see number of sequences per hdf5 file, total should add up to 34012
# later: merge those into one hdf5 file. also add embedddings 

file = sys.argv[1]
print(f'tfr file: {file}')

def make_parser(): 
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
    # tfr_path = f'/exports/humgen/idenhond/data/Basenji/tfrecords/{subset}*.tfr'
    # tfr_path = '/exports/humgen/idenhond/data/Basenji/tfrecords/train-0-0.tfr'
    tfr_path = file
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

targets = get_target() # numpy array, 34021 entries

filename = file.split('/')[-1].split('.')[0]

# if os.path.exists(f'/exports/humgen/idenhond/data/targets_train'): os.remove(f'hdf5test.hdf5')

with h5py.File(f'/exports/humgen/idenhond/data/targets_train/{filename}.hdf5', "a") as f:
    f.create_dataset('y', data = targets, compression="gzip", compression_opts=9)

with h5py.File(f'/exports/humgen/idenhond/data/targets_train/{filename}.hdf5') as p:
    print(f"Number of keys: {len(p.keys())}")
        # x = p['x'][()]
        # print(x.shape)
        # y = p['y'][()]
        # print(y.shape)
