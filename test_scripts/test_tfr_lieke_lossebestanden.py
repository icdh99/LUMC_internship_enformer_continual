import os
import tensorflow as tf
import numpy as np
from natsort import natsorted
import glob

# subset = 'test'
subset = 'valid'
# select tensorflow records for test sequences
tfr_path = f'/exports/humgen/idenhond/data/Basenji/tfrecords/{subset}*.tfr'
# tfr_path = '/exports/humgen/idenhond/Basenji_data/tfrecords/test*.tfr'
tfr_files = natsorted(glob.glob(tfr_path))
print(tfr_files)


dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
print(dataset)

# metadata human --> invullen in code
{
    "num_targets": 5313,
    "train_seqs": 34021,
    "valid_seqs": 2213,
    "test_seqs": 1937,
    "seq_length": 131072,
    "pool_width": 128,
    "crop_bp": 8192,
    "target_length": 896
}

# function to get info out of the records
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
        target = tf.reshape(target, (896, 5313))
        target = tf.cast(target, tf.float32)

        return sequence, target

    return parse_proto

def file_to_records(filename):
    return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

dataset = dataset.flat_map(file_to_records)
dataset = dataset.map(make_parser())
dataset = dataset.batch(1)
print(dataset)
print(type(dataset))

# initialize inputs and outputs
seqs_1hot = []
targets = []

# collect inputs and outputs
teller = 0
for seq_1hot, targets1 in dataset:
    # dataset for test sequences contains 1937 sets of sequences + targets
    # sequence
    print(teller)
    teller += 1
    seqs_1hot.append(seq_1hot.numpy())
    targets.append(targets1.numpy())
    print(f'shape sequence: {seq_1hot.numpy().shape}')
    print(f'shape target: {targets1.numpy().shape}')
    if teller == 2:
        break

# make arrays
seqs_1hot = np.array(seqs_1hot)
targets = np.array(targets)

print(seqs_1hot[0])
print(targets[0])
print(f'shape sequence: {seqs_1hot.shape}')
print(f'shape target: {targets.shape}')