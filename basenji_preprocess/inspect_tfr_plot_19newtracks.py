import os
import tensorflow as tf
from natsort import natsorted
import glob
import matplotlib.pyplot as plt

print(f'\nrunning inspect_tfr.py')
"""
This script is plotting some tracks from the human enformer tf records for a specified sequence. 

"""

def make_parser(num_tracks): #, rna_mode
    def parse_proto(example_protos):
        """Parse TFRecord protobuf."""

        feature_spec = {
        'sequence': tf.io.FixedLenFeature([], dtype = tf.string),  # Ignore this, resize our own bigger one
        'target': tf.io.FixedLenFeature([], dtype = tf.string),
        }

        # parse example into features
        feature_tensors = tf.io.parse_single_example(example_protos, features=feature_spec)

        sequence = tf.io.decode_raw(feature_tensors['sequence'], tf.bool)
        sequence = tf.reshape(sequence, (131072, 4))
        sequence = tf.cast(sequence, tf.float32)

        target = tf.io.decode_raw(feature_tensors['target'], tf.float16)
        target = tf.reshape(target, (896, num_tracks))
        target = tf.cast(target, tf.float32)

        return target, sequence

    return parse_proto

def file_to_records(filename):
    return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

# def get_target():
#     tfr_path = f'/exports/humgen/idenhond/data/basenji_preprocess/output_tfr/tfrecords/train-0.tfr'   # tfr records for new tracks (22 tracks stored in here)
#     tfr_files = natsorted(glob.glob(tfr_path))
#     print(f'number of tfr files: {len(tfr_files)}')
#     print(tfr_files)

#     NUM_TRACKS = 22
#     dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
#     dataset = dataset.flat_map(file_to_records)
#     dataset = dataset.map(make_parser(NUM_TRACKS))
#     dataset = dataset.batch(1)
#     print(type(dataset))

#     for i, (target, sequence) in enumerate(dataset):    # loop over sequences
#         print(i, target.shape)  # first sequence: chr18:936,578-1,051,266 in IGV
#         target = tf.squeeze(target)
#         print(target.shape)
#         plt.figure(figsize=(12, 4))
#         plt.plot(target[:, 1])  
#         plt.legend()
#         plt.title(f'new tracks train-0.tfr seq{i+1} track2_ENCFF914YXU')
#         plt.savefig(f'newtracks_train0_seq{i+1}_track2_ENCFF914YXU.png')
#         plt.close()
#         if i == 2:
#             break
#     return None
# get_target()

def get_target_mine_ENCFF967MGL():
    tfr_path = f'/exports/humgen/idenhond/data/basenji_preprocess/output_tfr/tfrecords/train-0.tfr'   # tfr records for new tracks (22 tracks stored in here)
    tfr_files = natsorted(glob.glob(tfr_path))
    print(f'number of tfr files: {len(tfr_files)}')
    print(tfr_files)

    NUM_TRACKS = 22
    dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
    dataset = dataset.flat_map(file_to_records)
    dataset = dataset.map(make_parser(NUM_TRACKS))
    dataset = dataset.batch(1)
    print(type(dataset))

    for i, (target, sequence) in enumerate(dataset):    # loop over sequences
        print(i, target.shape)  # first sequence: chr18:936,578-1,051,266 in IGV
        target = tf.squeeze(target)
        # print(target.shape)
        if i == 15:
            plt.figure(figsize=(12, 4))
            plt.plot(target[:, 6])  
            plt.legend()
            plt.title(f'new tracks train-0.tfr seq{i+1} track6_ENCFF967MGL')
            plt.savefig(f'newtracks_train0_seq{i+1}_track6_ENCFF967MGL.png')
            plt.close()
            break
        
    return None
get_target_mine_ENCFF967MGL()

def get_target_enformer_ENCFF318ORB():
    tfr_path = f'/exports/archive/hg-funcgenom-research/idenhond/Basenji/tfrecords/train-0-0.tfr'   
    tfr_files = natsorted(glob.glob(tfr_path))
    print(f'number of tfr files: {len(tfr_files)}')
    print(tfr_files)

    NUM_TRACKS = 5313
    dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
    dataset = dataset.flat_map(file_to_records)
    dataset = dataset.map(make_parser(NUM_TRACKS))
    dataset = dataset.batch(1)
    print(type(dataset))

    for i, (target, sequence) in enumerate(dataset):    # loop over sequences
        print(i, target.shape)  # first sequence: chr18:936,578-1,051,266 in IGV
        target = tf.squeeze(target)
        # print(target.shape)
        if i == 15:
            plt.figure(figsize=(12, 4))
            plt.plot(target[:, 4083])  
            plt.legend()
            plt.title(f'new tracks train-0.tfr seq{i+1} track4083_ENCFF318ORB')
            plt.savefig(f'newtracks_train0_seq{i+1}_track4083_ENCFF318ORB.png')
            plt.close()
            break
        
    return None
get_target_enformer_ENCFF318ORB()

