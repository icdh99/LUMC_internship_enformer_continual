import os
import tensorflow as tf
from natsort import natsorted
import glob

print(f'\nrunning inspect_tfr.py')

NUM_TRACKS = 5

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
    tfr_path = f'/exports/humgen/idenhond/data/basenji_preprocess/output_tfr/tfrecords/train-1.tfr'
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
        print(target)
        print(i, sequence.shape)
        print(sequence)
        break
    i = 0
    # for i, (target) in enumerate(dataset):
    #     i += 1

    print(f'i: {i}')
        # target_np = target.numpy()
        # print(type(target_np))
        # print(target_np.shape)
        # target_tensor = torch.from_numpy(target.numpy())
        # torch.save(target_tensor, f'/exports/humgen/idenhond/data/Enformer_train/Enformer_train_targets/targets_seq{i+1}.pt')
        # torch.save(target_tensor, f'/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_targets_perseq/targets_seq{i+1}.pt')
    return None

get_target()