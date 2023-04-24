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

def get_target_enformer():
    tfr_path = f'/exports/archive/hg-funcgenom-research/idenhond/Basenji/tfrecords/train-0-0.tfr'   # enformer tfr records from archive
    tfr_files = natsorted(glob.glob(tfr_path))
    print(f'number of tfr files: {len(tfr_files)}')
    print(tfr_files)

    NUM_TRACKS = 5313
    dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
    dataset = dataset.flat_map(file_to_records)
    dataset = dataset.map(make_parser(NUM_TRACKS))
    dataset = dataset.batch(1)
    print(type(dataset))

    track_nr = 1028
    for i, (target, sequence) in enumerate(dataset):    # loop over sequences
        print(i, target.shape)  # first sequence: chr2:140,433,983-140,548,671 in IGV
        target = tf.squeeze(target)
        print(target.shape)
        print(f'track 1 enformer:')
        print(target[:, 0][:10])
        print(f'track 2 enformer:')
        print(target[:, 1454][:10])
        print(f'track 3 enformer:')
        print(target[:, 1028][:10])
        # plt.figure(figsize=(12, 4))
        # plt.plot(target[:, track_nr])  
        # plt.legend()
        # plt.title(f'enformer train-0.tfr seq{i+1} track1')
        # plt.savefig(f'plots_human_enformer/human_enformer_train0_seq{i+1}_track4_TF.png')
        # plt.close()
        if i == 0:
            break
    return None
get_target_enformer()

def get_target_mine():
    tfr_path = f'/exports/humgen/idenhond/data/basenji_preprocess/output_tfr/tfrecords/train-0.tfr' # new tracks
    tfr_files = natsorted(glob.glob(tfr_path))
    print(f'number of tfr files: {len(tfr_files)}')
    print(tfr_files)

    NUM_TRACKS = 22
    dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
    dataset = dataset.flat_map(file_to_records)
    dataset = dataset.map(make_parser(NUM_TRACKS))
    dataset = dataset.batch(1)
    print(type(dataset))

    track_nr = 4
    for i, (target, sequence) in enumerate(dataset):
        print(i, target.shape)  
        target = tf.squeeze(target)
        print(target.shape)
        print(f'track 1 mine:')
        print(target[:, 2][:10])
        print(f'track 2 mine:')
        print(target[:, 3][:10])
        print(f'track 3 mine:')
        print(target[:, 4][:10])
        # plt.figure(figsize=(12, 4))
        # plt.plot(target[:, track_nr])  
        # plt.legend()
        # plt.title(f'mine train-0.tfr seq{i+1} track1')
        # plt.savefig(f'plots_human_enformer/human_enformer_mytarget_train0_seq{i+1}_track4_TF.png')
        # plt.close()
        if i == 0:
            break

    return None
get_target_mine()