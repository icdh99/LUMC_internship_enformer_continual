import os
import tensorflow as tf
from natsort import natsorted
import glob
import matplotlib.pyplot as plt

print(f'\nrunning inspect_tfr.py')

NUM_TRACKS = 13

def make_parser(): #, rna_mode
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
        target = tf.reshape(target, (896, NUM_TRACKS))
        target = tf.cast(target, tf.float32)

        return target, sequence

    return parse_proto

def file_to_records(filename):
    return tf.data.TFRecordDataset(filename, compression_type='ZLIB')


def get_target_scale4():
    tfr_path = f'/exports/humgen/idenhond/data/basenji_preprocess/output_tfr_mouse_snatac_scale4/tfrecords/train-0.tfr'
    tfr_files = natsorted(glob.glob(tfr_path))
    print(f'number of tfr files: {len(tfr_files)}')
    print(tfr_files)
    dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
    dataset = dataset.flat_map(file_to_records)
    dataset = dataset.map(make_parser())
    dataset = dataset.batch(1)
    print(type(dataset))

    for i, (target, sequence) in enumerate(dataset):
        print(i, target.shape)
        target = tf.squeeze(target)
        print(target[:, 0]) # remove first dimension to get shape (896, 1)
        plt.plot(target[:, 0], label = 'scale 4')
        plt.savefig(f'train_seq{i+1}_track1_scale4.png')
        plt.close()
        if i == 1:
            break
    return None

get_target_scale4()

def get_target_scale2():
    tfr_path = f'/exports/humgen/idenhond/data/basenji_preprocess/output_tfr_mouse_snatac_scale2/tfrecords/train-0.tfr'
    tfr_files = natsorted(glob.glob(tfr_path))
    print(f'number of tfr files: {len(tfr_files)}')
    print(tfr_files)
    dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
    dataset = dataset.flat_map(file_to_records)
    dataset = dataset.map(make_parser())
    dataset = dataset.batch(1)
    print(type(dataset))

    for i, (target, sequence) in enumerate(dataset):
        print(i, target.shape)
        target = tf.squeeze(target)
        print(target[:, 0]) # remove first dimension to get shape (896, 1)
        plt.plot(target[:, 0], label = 'scale 2')
        plt.legend()
        plt.savefig(f'train_seq{i+1}_track1_scale2.png')
        plt.close()
        if i == 1:
            break

    return None

get_target_scale2()

def get_target_scale1():
    tfr_path = f'/exports/humgen/idenhond/data/basenji_preprocess/output_tfr_mouse_snatac_scale1/tfrecords/train-0.tfr'
    tfr_files = natsorted(glob.glob(tfr_path))
    print(f'number of tfr files: {len(tfr_files)}')
    print(tfr_files)
    dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
    dataset = dataset.flat_map(file_to_records)
    dataset = dataset.map(make_parser())
    dataset = dataset.batch(1)
    print(type(dataset))

    for i, (target, sequence) in enumerate(dataset):
        print(i, target.shape)
        target = tf.squeeze(target)
        print(target[:, 0]) # remove first dimension to get shape (896, 1)
        # top_values, top_indices = tf.math.top_k(target[:, 0], k=5)
        # print("Top values:", top_values.numpy())
        # print("Indices:", top_indices.numpy())
        plt.plot(target[:, 0], label = 'scale 1')
        plt.legend()
        plt.savefig(f'train_seq{i+1}_track1_scale1.png')
        plt.close()

        if i == 1:
            break
    
    
    return None

get_target_scale1()