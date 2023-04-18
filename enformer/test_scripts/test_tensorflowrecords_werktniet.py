import numpy as np
import tensorflow as tf

import os 
import json
import pandas as pd
import torch
import pyfaidx

import kipoiseq
import functools
from kipoiseq import Interval


file_name = '/exports/humgen/idenhond/data_human_tfrecords_test-0-0.tfr'
filename = '/exports/humgen/idenhond/enformer_dev/enformer-pytorch/test-0-0.tfr'
filename = '/exports/humgen/idenhond/Basenji_data/tfrecords/test-0-7.tfr'
# print(sum(1 for _ in tf.io.tf_record_iterator(file_name)))


# for example in tf.io.tf_record_iterator("/exports/humgen/idenhond/data_human_tfrecords_test-0-0.tfr"):
#     print(tf.train.Example.FromString(example))
tf.executing_eagerly()
sum(1 for _ in tf.data.TFRecordDataset(file_name))
exit()

# from https://stackoverflow.com/questions/52099863/how-to-read-decode-tfrecords-with-tf-data-api

"""
def _read_from_tfrecord(example_proto):
    feature = {
        'sequence': tf.io.FixedLenFeature([], tf.string),  # Ignore this, resize our own bigger one
        'target': tf.io.FixedLenFeature([], tf.string),
    }

    features = tf.io.parse_example([example_proto], features=feature)

    # Since the arrays were stored as strings, they are now 1d 
    sequence = tf.io.decode_raw(features['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (196_608, 4))
    sequence = tf.cast(sequence, tf.float32)

    # sample_1d = tf.io.decode_raw(features['target'], tf.int64)
    target = tf.io.decode_raw(features['target'], tf.float16)
    target = tf.reshape(target,
                        (896, 5313))
    target = tf.cast(target, tf.float32)


    # In order to make the arrays in their original shape, they have to be reshaped.
    # label_restored = tf.reshape(label_1d, tf.stack([2, 3, -1]))
    # sample_restored = tf.reshape(sample_1d, tf.stack([2, 3, -1]))

    return sequence, target


filenames = ['/exports/humgen/idenhond/data_human_tfrecords_test-0-1.tfr']
dataset = tf.data.TFRecordDataset(filenames)
print(dataset)
dataset = dataset.map(_read_from_tfrecord)
print(type(dataset))
print(dataset.output_types)
print(dataset.output_shapes)
exit()
iterator = tf.data.Iterator.from_structure(dataset.output_types, dataset.output_shapes)
label_tf, sample_tf = iterator.get_next()
iterator_init = iterator.make_initializer(dataset, name="dataset_init")

with tf.Session() as sess:
    # Initialise the iterator passing the name of the TFRecord file to the placeholder
    sess.run(iterator_init, feed_dict={data_path: filename})

    # Obtain the images and labels back
    read_label, read_sample = sess.run([label_tf, sample_tf])
"""

# from https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset
# Read the data back out.

def decode_fn(record_bytes):
  return tf.io.parse_single_example(
      # Data
      record_bytes,
      # Schema
      {
        'sequence': tf.io.FixedLenFeature([], dtype = tf.string),  # Ignore this, resize our own bigger one
        'target': tf.io.FixedLenFeature([], dtype = tf.string),
    }

  )

example_path = '/exports/humgen/idenhond/data_human_tfrecords_test-0-0.tfr'
for batch in tf.data.TFRecordDataset([example_path]).map(decode_fn):
    print(x, y)
    break


# from enformer notebook
def deserialize(serialized_example, metadata):
    """Deserialize bytes stored in TFRecordFile."""
    # Deserialization
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),  # Ignore this, resize our own bigger one
        'target': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_example(serialized_example, feature_map)
    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
    sequence = tf.cast(sequence, tf.float32)

    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(target,
                        (metadata['target_length'], metadata['num_targets']))
    target = tf.cast(target, tf.float32)

    return {'sequence_old': sequence,
            'target': target}