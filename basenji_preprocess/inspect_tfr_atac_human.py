import os
import tensorflow as tf
from natsort import natsorted
import glob
import matplotlib.pyplot as plt

print(f'\nrunning inspect_tfr.py')

NUM_TRACKS = 66

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


def get_target():
    tfr_path = f'/exports/humgen/idenhond/data/basenji_preprocess/output_tfr_human_atac/tfrecords/train-0.tfr'
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
        # print(target[:, 0]) # remove first dimension to get shape (896, 1)
        plt.figure(figsize=(12, 4))
        plt.plot(target[:, 22], label = 'track 22')
        plt.title(f'human atac tracks train-0.tfr seq{i+1} track 22 Human_ATAC_GABAergic')
        plt.savefig(f'plots_human_atac/humanatac-train_seq{i+1}_track22_Human_ATAC_GABAergic.png')
        plt.close()

        plt.figure(figsize=(12, 4))
        plt.plot(target[:, 55], label = 'track 22')
        plt.title(f'human atac tracks train-0.tfr seq{i+1} track 55 Human_ATAC_Oligo')
        plt.savefig(f'plots_human_atac/humanatac-train_seq{i+1}_track55_Human_ATAC_Oligo.png')
        plt.close()
        if i == 1:
            break
    return None

get_target()



