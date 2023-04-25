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
import sys

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device (set to GPU if available):', device)

subset = sys.argv[1]
print(f'subset: {subset}')

def make_parser(): #, rna_mode
    def parse_proto(example_protos):

        NUM_TRACKS = 27
        """Parse TFRecord protobuf."""
        feature_spec = {
        'sequence': tf.io.FixedLenFeature([], dtype = tf.string),
        'target': tf.io.FixedLenFeature([], dtype = tf.string),
        }
        feature_tensors = tf.io.parse_single_example(example_protos, features=feature_spec)

        sequence = tf.io.decode_raw(feature_tensors['sequence'], tf.bool)
        sequence = tf.reshape(sequence, (131072, 4))
        sequence = tf.cast(sequence, tf.float32)

        target = tf.io.decode_raw(feature_tensors['target'], tf.float16)
        target = tf.reshape(target, (896, NUM_TRACKS))
        target = tf.cast(target, tf.float32)
        return target, sequence
    return parse_proto

def file_to_records(filename):
    return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

def get_target(subset = subset): 
    if subset == 'train':
        tfr_path = f'/exports/humgen/idenhond/data/basenji_preprocess/output_tfr_27newtracks_human/tfrecords/train*.tfr'
    if subset == 'validation':
        tfr_path = f'/exports/humgen/idenhond/data/basenji_preprocess/output_tfr_27newtracks_human/tfrecords/valid*.tfr'
    if subset == 'test':
        tfr_path = f'/exports/humgen/idenhond/data/basenji_preprocess/output_tfr_27newtracks_human/tfrecords/test*.tfr'
    tfr_files = natsorted(glob.glob(tfr_path))
    print(f'number of tfr files for {subset} subset: {len(tfr_files)}')
    dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
    dataset = dataset.flat_map(file_to_records)
    dataset = dataset.map(make_parser())
    dataset = dataset.batch(1)

    # instead of adding targets to big array, save to seperate file in iteration
    # for i, target in enumerate(dataset):
    for i, (target, sequence) in enumerate(dataset):
        print(i)
        target_tensor = torch.from_numpy(target.numpy())
        print(target_tensor.shape)
        torch.save(target_tensor, f'/exports/humgen/idenhond/data/Enformer_{subset}/Newtracks_2404_{subset}_targets/targets_seq{i+1}.pt')
    return None

get_target()