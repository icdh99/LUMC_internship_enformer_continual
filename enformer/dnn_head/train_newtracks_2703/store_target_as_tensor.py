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

def make_parser(): #, rna_mode
    def parse_proto(example_protos):
        """Parse TFRecord protobuf."""
        feature_spec = {
        'sequence': tf.io.FixedLenFeature([], dtype = tf.string),
        'target': tf.io.FixedLenFeature([], dtype = tf.string),
        }
        feature_tensors = tf.io.parse_single_example(example_protos, features=feature_spec)
        target = tf.io.decode_raw(feature_tensors['target'], tf.float16)
        target = tf.reshape(target, (896, 22))                  ### 22 tracks
        target = tf.cast(target, tf.float32)
        return target
    return parse_proto

def file_to_records(filename):
    return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

def get_target(subset = 'train'):
    tfr_path = f'/exports/humgen/idenhond/data/basenji_preprocess/output_tfr/tfrecords/{subset}*.tfr'
    tfr_files = natsorted(glob.glob(tfr_path))
    print(f'number of tfr files for {subset} subset: {len(tfr_files)}')
    dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
    dataset = dataset.flat_map(file_to_records)
    dataset = dataset.map(make_parser())
    dataset = dataset.batch(1)

    # instead of adding targets to big array, save to seperate file in iteration
    # save to folder data/Enformer_train/Enformer_train_targets
    # targets = []
    for i, target in enumerate(dataset):
        # # target_np = target.numpy()
        # # print(type(target_np))
        # # print(target_np.shape)
        # target_tensor = torch.from_numpy(target.numpy())
        # # print(target_tensor.shape)
        # new_tensor = torch.zeros(1,896,19)
        # indices = [0, 1,2,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
        # for x, j in enumerate(indices):
        #     new_tensor[:,:,x] = target_tensor[:,:, j]

        # # print(new_tensor.shape)

        # # torch.save(new_tensor, f'/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_targets_newtracks2703/targets_seq{i+1}.pt')
        # # torch.save(new_tensor, f'/exports/humgen/idenhond/data/Enformer_test/Enformer_test_targets_newtracks2703/targets_seq{i+1}.pt')
        # torch.save(new_tensor, f'/exports/humgen/idenhond/data/Enformer_train/Enformer_train_targets_newtracks2703/targets_seq{i+1}.pt')
        # hierboven was hoe je de targets voor de new tracks only hebt opgeslage (dus verkeerd, je moet 2,3,4 overslaan bij de indices en niet 3,4,5)

        # nu hier: opslaan alleen oude targets uit de 22 tracks
        target_tensor = torch.from_numpy(target.numpy())
        new_tensor = torch.zeros(1,896,3)
        indices = [2,3,4]
        for x, j in enumerate(indices):
            new_tensor[:,:,x] = target_tensor[:,:, j]

        torch.save(new_tensor, f'/exports/humgen/idenhond/data/Enformer_train/Enformer_train_3tracks_remade/targets_seq{i+1}.pt')

    return None

get_target()



"""
de tfr files hebben 22 tracks (index 0 - 21)
index 3,4,5 zijn van enformer --> die moet je eruit halen

"""