import os
import tensorflow as tf
from natsort import natsorted
import glob
import matplotlib.pyplot as plt

def make_parser_mine(): #, rna_mode
    def parse_proto(example_protos):
        feature_spec = {'sequence': tf.io.FixedLenFeature([], dtype = tf.string), 'target': tf.io.FixedLenFeature([], dtype = tf.string)}
        feature_tensors = tf.io.parse_single_example(example_protos, features=feature_spec)
        sequence = tf.io.decode_raw(feature_tensors['sequence'], tf.bool)
        sequence = tf.reshape(sequence, (131072, 4))
        sequence = tf.cast(sequence, tf.float32)
        target = tf.io.decode_raw(feature_tensors['target'], tf.float16)
        target = tf.reshape(target, (896, 22))
        target = tf.cast(target, tf.float32)
        return target, sequence
    return parse_proto

def file_to_records_mine(filename):
    return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

def get_target_mine():
    tfr_path = f'/exports/humgen/idenhond/data/basenji_preprocess/output_tfr/tfrecords/test-0.tfr'
    tfr_files = natsorted(glob.glob(tfr_path))
    print(f'number of tfr files: {len(tfr_files)}')
    print(tfr_files)
    dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
    dataset = dataset.flat_map(file_to_records_mine)
    dataset = dataset.map(make_parser_mine())
    dataset = dataset.batch(1)


    i = 0
    return dataset

dataset_mine = get_target_mine()
# for i, (target, sequence) in enumerate(dataset_mine):
#     # print(i, target.shape)
#     # print(target)
#     target_mine = target
#     # print(i, sequence.shape)
#     # print(sequence)
#     seq_mine = sequence
#     break
# print(target_mine[:, :, 2])
# print(target_mine[:, :, 2].shape)
# print(seq_mine)


def make_parser(): #, rna_mode
    def parse_proto(example_protos):
        feature_spec = {'sequence': tf.io.FixedLenFeature([], dtype = tf.string), 'target': tf.io.FixedLenFeature([], dtype = tf.string)}
        feature_tensors = tf.io.parse_single_example(example_protos, features=feature_spec)
        sequence = tf.io.decode_raw(feature_tensors['sequence'], tf.bool)
        sequence = tf.reshape(sequence, (131072, 4))
        sequence = tf.cast(sequence, tf.float32)
        target = tf.io.decode_raw(feature_tensors['target'], tf.float16)
        target = tf.reshape(target, (896, 5313))
        target = tf.cast(target, tf.float32)
        return target, sequence
    return parse_proto

def file_to_records(filename):
    return tf.data.TFRecordDataset(filename, compression_type='ZLIB')

def get_target():
    tfr_path = f'/exports/humgen/idenhond/data/Basenji/tfrecords/test-0-0.tfr'
    tfr_files = natsorted(glob.glob(tfr_path))
    print(f'number of tfr files: {len(tfr_files)}')
    print(tfr_files)
    dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
    dataset = dataset.flat_map(file_to_records)
    dataset = dataset.map(make_parser())
    dataset = dataset.batch(1)

    # for i, (target, sequence) in enumerate(dataset):
    #     # print(i, target.shape)
    #     # print(target)
    #     target = target
    #     # print(i, sequence.shape)
    #     # print(sequence)
    #     seq = sequence
    #     break

    return dataset

dataset = get_target()
for i, ((target, sequence), (target_mine, sequence_mine)) in enumerate(zip(dataset, dataset_mine)):
    # print(i, target.shape)
    # print(target)
    target = target
    # print(i, sequence.shape)
    # print(sequence)
    seq = sequence
    target_mine = target_mine
    seq_mine = sequence_mine
    print(tf.equal(seq, seq_mine))
    print(i)

    # difference between tracks
    plt.figure()
    plt.plot(tf.squeeze(tf.subtract(target[:, :, 0],target_mine[:, :, 2])), label = 'track 2')
    plt.legend()
    plt.title(f'seq {i}')
    plt.savefig(f'plots/seq{i}_track2_diff.png')
    plt.close()

    plt.figure()
    plt.plot(tf.squeeze(tf.subtract(target[:, :, 1454],target_mine[:, :, 3])), label = 'track 3')
    plt.legend()
    plt.title(f'seq {i}')
    plt.savefig(f'plots/seq{i}_track3_diff.png')
    plt.close()
    
    plt.figure()
    plt.plot(tf.squeeze(tf.subtract(target[:, :, 1028],target_mine[:, :, 4])), label = 'track 4')
    plt.legend()
    plt.title(f'seq {i}')
    plt.savefig(f'plots/seq{i}_track4_diff.png')
    plt.close()
    
    # plot both tracks
    plt.figure()
    plt.plot(tf.squeeze(target_mine[:, :, 2]), label = 'track 2 mine')
    plt.plot(tf.squeeze(target[:, :, 0]), label = 'track 2 enformer', linewidth = 0.5)
    # print(target[:, :, 0]).shape)
    # print(tf.squeeze((target[:, :, 0]).shape))
    plt.legend()
    plt.title(f'seq {i}')
    plt.savefig(f'plots/seq{i}_track2.png')
    plt.close()

    plt.figure()
    plt.plot(tf.squeeze(target_mine[:, :, 3]), label = 'track 3 mine')
    plt.plot(tf.squeeze(target[:, :, 1454]), label = 'track 3 enformer', linewidth = 0.5)
    plt.legend()
    plt.title(f'seq {i}')
    plt.savefig(f'plots/seq{i}_track3.png')
    plt.close()
    
    plt.figure()
    plt.plot(tf.squeeze(target_mine[:, :, 4]), label = 'track 4 mine')
    plt.plot(tf.squeeze(target[:, :, 1028]), label = 'track 4 enformer', linewidth = 0.5)
    plt.legend()
    plt.title(f'seq {i}')
    plt.savefig(f'plots/seq{i}_track4.png')
    plt.close()
    if i == 5:
        break
# print(target[:, :, 0])
# print(target[:, :, 0].shape)
# print(seq)

# 1e sequence
# print(tf.equal(target[:, :, 0], target_mine[:, :, 2])) # cerebellum
# print(tf.equal(target[:, :, 1454], target_mine[:, :, 3]))
# print(tf.equal(target[:, :, 1028], target_mine[:, :, 4]))
# print(tf.equal(target[:, :, 1028] - target_mine[:, :, 4]))
# print(tf.subtract(target[:, :, 1028],target_mine[:, :, 4]))
# print(tf.squeeze(tf.subtract(target[:, :, 1028],target_mine[:, :, 4])).shape)

# print(tf.equal(seq, seq_mine))

# plt.figure()
# plt.plot(tf.squeeze(tf.subtract(target[:, :, 0],target_mine[:, :, 2])), label = 'track 2')
# plt.legend()
# plt.savefig('track2.png')
# plt.figure()
# plt.plot(tf.squeeze(tf.subtract(target[:, :, 1454],target_mine[:, :, 3])), label = 'track 3')
# plt.legend()
# plt.savefig('track3.png')
# plt.figure()
# plt.plot(tf.squeeze(tf.subtract(target[:, :, 1028],target_mine[:, :, 4])), label = 'track 4')
# plt.legend()
# plt.savefig('track4.png')


"""
target mine
0 'ENCFF601VTB ENCFF601VTB '32 '2 'mean 'brain_tissue_female_embryo'])  # histone chip encode 
1 'ENCFF914YXU ENCFF914YXU '32 '2 'mean 'liver_tissue'])   # histone chip encode
2 'ENCFF833POA ENCFF833POA '32 '2 'mean 'DNASE:cerebellum male adult (27 years) and male adult (35 years)']) --> enformer target: index 0
3 'ENCFF828RQS ENCFF828RQS '32 '2 'mean 'CHIP:H3K9me3:stomach smooth muscle female adult (84 years)']) --> enformer table index 1454
4 'ENCFF003HJB ENCFF003HJB '32 '2 'mean 'CHIP:CEBPB:HepG2']) --> enformer table index 1028
"""

