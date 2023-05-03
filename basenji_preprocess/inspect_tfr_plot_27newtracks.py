import os
import tensorflow as tf
from natsort import natsorted
import glob
import matplotlib.pyplot as plt
import torch

print(f'\nrunning inspect_tfr.py')
"""
This script is plotting some tracks from the human enformer tf records for a specified sequence. 

Seq 1 van train met crop: chr18:936,578-1,051,266 
seq 2 van train met crop: chr4:113,639,139-113,753,827
Seq 3 van train met crop: chr11:18,435,912-18,550,600

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
        target = tf.reshape(target, (896, 27))
        target = tf.cast(target, tf.float32)

        return target, sequence

    return parse_proto

def file_to_records(filename):
    return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

"""
26/4: plot ENCFF194XNN track from targets. This is the histone track with a very low correlation in the prediction of the test set.
"""
def get_target():
    tfr_path = f'/exports/humgen/idenhond/data/basenji_preprocess/output_tfr_27newtracks_human/tfrecords/test-0.tfr'   # tfr records for new tracks (22 tracks stored in here)
    tfr_files = natsorted(glob.glob(tfr_path))
    print(f'number of tfr files: {len(tfr_files)}')
    print(tfr_files)

    NUM_TRACKS = 27
    dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
    dataset = dataset.flat_map(file_to_records)
    dataset = dataset.map(make_parser(NUM_TRACKS))
    dataset = dataset.batch(1)
    print(type(dataset))

    for i, (target, sequence) in enumerate(dataset):    # loop over sequences
        print(i, target.shape) 
        target = tf.squeeze(target)
        print(target.shape)
        plt.figure(figsize=(12, 4))
        plt.plot(target[:, 4])  
        plt.legend()
        plt.title(f'new tracks 27 test-0.tfr seq{i+1} track4_ENCFF194XNN')
        plt.savefig(f'newtracks27_test0_seq{i+1}_track4_ENCFF194XNN.png')
        plt.close()
        if i == 2:
            break
    return None
get_target()

"""
26/4: plot test predictions for the ENCFF194XNN track. 
"""

for i in range(3):
    print(i)
    output_seq = torch.load(f'/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output_newtracks_2404/output_seq{i+1}.pt', map_location=torch.device('cpu')).squeeze()
    plt.figure(figsize=(12, 4))
    print(output_seq.shape)
    plt.plot(output_seq[:, 4])
    plt.savefig(f'newtracks27_prediction_test0_seq{i+1}_track4_ENCFF194XNN.png', bbox_inches='tight')
    


"""
26/4: plot ENCFF515FSI track from targets. Histone track with correlation 0.60
"""
def get_target():
    tfr_path = f'/exports/humgen/idenhond/data/basenji_preprocess/output_tfr_27newtracks_human/tfrecords/test-0.tfr'   # tfr records for new tracks (22 tracks stored in here)
    tfr_files = natsorted(glob.glob(tfr_path))
    print(f'number of tfr files: {len(tfr_files)}')
    print(tfr_files)

    NUM_TRACKS = 27
    dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
    dataset = dataset.flat_map(file_to_records)
    dataset = dataset.map(make_parser(NUM_TRACKS))
    dataset = dataset.batch(1)
    print(type(dataset))

    for i, (target, sequence) in enumerate(dataset):    # loop over sequences
        print(i, target.shape) 
        target = tf.squeeze(target)
        print(target.shape)
        plt.figure(figsize=(12, 4))
        plt.plot(target[:, 8])  
        plt.legend()
        plt.title(f'new tracks 27 test-0.tfr seq{i+1} track8 ENCFF515FSI')
        plt.savefig(f'newtracks27_test0_seq{i+1}_track8_ENCFF515FSI.png')
        plt.close()
        if i == 2:
            break
    return None
get_target()

"""
26/4: plot test predictions for the ENCFF515FSI track. 
"""

for i in range(3):
    print(i)
    output_seq = torch.load(f'/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output_newtracks_2404/output_seq{i+1}.pt', map_location=torch.device('cpu')).squeeze()
    plt.figure(figsize=(12, 4))
    print(output_seq.shape)
    plt.plot(output_seq[:, 8])
    plt.title(f'new tracks 27 predictions test-0.tfr seq{i+1} track8 ENCFF515FSI')
    plt.savefig(f'newtracks27_prediction_test0_seq{i+1}_track8_ENCFF515FSI.png', bbox_inches='tight')
    


"""
26/4: plot ENCFF499NHY track from targets. DNASE track with correlation 0.72
"""
def get_target():
    tfr_path = f'/exports/humgen/idenhond/data/basenji_preprocess/output_tfr_27newtracks_human/tfrecords/test-0.tfr'   # tfr records for new tracks (22 tracks stored in here)
    tfr_files = natsorted(glob.glob(tfr_path))
    print(f'number of tfr files: {len(tfr_files)}')
    print(tfr_files)

    NUM_TRACKS = 27
    dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
    dataset = dataset.flat_map(file_to_records)
    dataset = dataset.map(make_parser(NUM_TRACKS))
    dataset = dataset.batch(1)
    print(type(dataset))

    for i, (target, sequence) in enumerate(dataset):    # loop over sequences
        print(i, target.shape) 
        target = tf.squeeze(target)
        print(target.shape)
        plt.figure(figsize=(12, 4))
        plt.plot(target[:, 8])  
        plt.legend()
        plt.title(f'new tracks 27 test-0.tfr seq{i+1} track22 ENCFF499NHY')
        plt.savefig(f'newtracks27_test0_seq{i+1}_track22_ENCFF499NHY.png')
        plt.close()
        if i == 2:
            break
    return None
get_target()

"""
26/4: plot test predictions for the ENCFF499NHY track. 
"""

for i in range(3):
    print(i)
    output_seq = torch.load(f'/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output_newtracks_2404/output_seq{i+1}.pt', map_location=torch.device('cpu')).squeeze()
    plt.figure(figsize=(12, 4))
    print(output_seq.shape)
    plt.plot(output_seq[:, 8])
    plt.title(f'new tracks 27 predictions test-0.tfr seq{i+1} track22 ENCFF499NHY')
    plt.savefig(f'newtracks27_prediction_test0_seq{i+1}_track22_ENCFF499NHY.png', bbox_inches='tight')
    