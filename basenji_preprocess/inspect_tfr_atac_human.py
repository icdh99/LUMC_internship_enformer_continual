import os
# import tensorflow as tf
from natsort import natsorted
import glob
import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns

print(f'\nrunning inspect_tfr.py')

NUM_TRACKS = 66

# def make_parser(): #, rna_mode
#     def parse_proto(example_protos):
#         """Parse TFRecord protobuf."""

#         feature_spec = {
#         'sequence': tf.io.FixedLenFeature([], dtype = tf.string),  # Ignore this, resize our own bigger one
#         'target': tf.io.FixedLenFeature([], dtype = tf.string),
#         }

#         # parse example into features
#         feature_tensors = tf.io.parse_single_example(example_protos, features=feature_spec)

#         sequence = tf.io.decode_raw(feature_tensors['sequence'], tf.bool)
#         sequence = tf.reshape(sequence, (131072, 4))
#         sequence = tf.cast(sequence, tf.float32)

#         target = tf.io.decode_raw(feature_tensors['target'], tf.float16)
#         target = tf.reshape(target, (896, NUM_TRACKS))
#         target = tf.cast(target, tf.float32)

#         return target, sequence

#     return parse_proto

# def file_to_records(filename):
#     return tf.data.TFRecordDataset(filename, compression_type='ZLIB')


# def get_target():
#     tfr_path = f'/exports/humgen/idenhond/data/basenji_preprocess/output_tfr_human_atac/tfrecords/train-0.tfr'
#     tfr_files = natsorted(glob.glob(tfr_path))
#     print(f'number of tfr files: {len(tfr_files)}')
#     print(tfr_files)
#     dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
#     dataset = dataset.flat_map(file_to_records)
#     dataset = dataset.map(make_parser())
#     dataset = dataset.batch(1)
#     print(type(dataset))

#     for i, (target, sequence) in enumerate(dataset):
#         print(i, target.shape)
#         target = tf.squeeze(target)
#         # print(target[:, 0]) # remove first dimension to get shape (896, 1)
#         plt.figure(figsize=(12, 4))
#         plt.plot(target[:, 22], label = 'track 22')
#         plt.title(f'human atac tracks train-0.tfr seq{i+1} track 22 Human_ATAC_GABAergic')
#         plt.savefig(f'plots_human_atac/humanatac-train_seq{i+1}_track22_Human_ATAC_GABAergic.png')
#         plt.close()

#         plt.figure(figsize=(12, 4))
#         plt.plot(target[:, 55], label = 'track 22')
#         plt.title(f'human atac tracks train-0.tfr seq{i+1} track 55 Human_ATAC_Oligo')
#         plt.savefig(f'plots_human_atac/humanatac-train_seq{i+1}_track55_Human_ATAC_Oligo.png')
#         plt.close()
#         if i == 1:
#             break
#     return None

# get_target()


# def get_target():
#     tfr_path = f'/exports/humgen/idenhond/data/basenji_preprocess/output_tfr_human_atac_normalized/tfrecords/train-0.tfr'
#     tfr_files = natsorted(glob.glob(tfr_path))
#     print(f'number of tfr files: {len(tfr_files)}')
#     print(tfr_files)
#     dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
#     dataset = dataset.flat_map(file_to_records)
#     dataset = dataset.map(make_parser())
#     dataset = dataset.batch(1)
#     print(type(dataset))

#     for i, (target, sequence) in enumerate(dataset):
#         print(i, target.shape)
#         target = tf.squeeze(target)
#         # print(target[:, 0]) # remove first dimension to get shape (896, 1)
#         plt.figure(figsize=(12, 4))
#         plt.plot(target[:, 22], label = 'track 22')
#         plt.title(f'human atac tracks train-0.tfr seq{i+1} track 22 Human_ATAC_GABAergic')
#         plt.savefig(f'plots_human_atac/humanatac_normalized-train_seq{i+1}_track22_Human_ATAC_GABAergic.png')
#         plt.close()

#         plt.figure(figsize=(12, 4))
#         plt.plot(target[:, 55], label = 'track 55')
#         plt.title(f'human atac tracks train-0.tfr seq{i+1} track 55 Human_ATAC_Oligo')
#         plt.savefig(f'plots_human_atac/humanatac_normalized-train_seq{i+1}_track55_Human_ATAC_Oligo.png')
#         plt.close()
#         if i == 1:
#             break
#     return None

# get_target()

"""
26/4: plot predictions for track 55 + 22 for train sequence 1 + 2
"""
# for i in range(2):
#     print(i)
#     output_seq = torch.load(f'/exports/humgen/idenhond/data/Enformer_train/Enformer_train_output_humanatac/output_seq{i+1}.pt', map_location=torch.device('cpu')).squeeze()
#     plt.figure(figsize=(12, 4))
#     print(output_seq.shape)
#     plt.plot(output_seq[:, 22])
#     plt.title(f'human atac tracks predictions train-0.tfr seq{i+1} track 22 Human_ATAC_GABAergic')
#     plt.savefig(f'plots_human_atac/humanatac_prediction_train0_seq{i+1}_track22_Human_ATAC_GABAergic.png', bbox_inches='tight')
    

#     plt.figure(figsize=(12, 4))
#     print(output_seq.shape)
#     plt.plot(output_seq[:, 55])
#     plt.title(f'human atac tracks predictions train-0.tfr seq{i+1} track 22 Human_ATAC_Oligo')
#     plt.savefig(f'plots_human_atac/humanatac_prediction_train0_seq{i+1}_track55_Human_ATAC_Oligo.png', bbox_inches='tight')
#     plt.close()
    

"""
26/4: plot tracks like in paper
"""
# def get_target():
#     tfr_path = f'/exports/humgen/idenhond/data/basenji_preprocess/output_tfr_human_atac/tfrecords/train-0.tfr'
#     tfr_files = natsorted(glob.glob(tfr_path))
#     print(f'number of tfr files: {len(tfr_files)}')
#     print(tfr_files)
#     dataset = tf.data.Dataset.from_tensor_slices(tfr_files)
#     dataset = dataset.flat_map(file_to_records)
#     dataset = dataset.map(make_parser())
#     dataset = dataset.batch(1)
#     print(type(dataset))

#     for i, (target, sequence) in enumerate(dataset):
#         print(i, target.shape)
#         target = tf.squeeze(target)
#         # target_22 = target[:, 22]
#         # target_55 = target[:, 55]
#         # print(target[:, 0]) # remove first dimension to get shape (896, 1)
#         # plt.figure(figsize=(12, 4))
#         # plt.plot(target[:, 22], label = 'track 22')
#         # plt.title(f'human atac tracks train-0.tfr seq{i+1} track 22 Human_ATAC_GABAergic')
#         # plt.savefig(f'plots_human_atac/humanatac-train_seq{i+1}_track22_Human_ATAC_GABAergic.png')
#         # plt.close()

#         # plt.figure(figsize=(12, 4))
#         # plt.plot(target[:, 55], label = 'track 22')
#         # plt.title(f'human atac tracks train-0.tfr seq{i+1} track 55 Human_ATAC_Oligo')
#         # plt.savefig(f'plots_human_atac/humanatac-train_seq{i+1}_track55_Human_ATAC_Oligo.png')
#         # plt.close()
#         if i == 0:
#             break
#     return target

# target = get_target()

# def plot_tracks(tracks, interval, name, height=1.5):
#   fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
#   for i, (ax, (title, y)) in enumerate(zip(axes, tracks.items())):
#     print(i)
#     if i == 0 or i == 1 or i == 2:
#         color = 'steelblue'
#     else:
#         color = 'sandybrown'
  
#     ax.fill_between(np.linspace(936578, 1051266, num=len(y)), y, color = color) 
#     # chr18:936,578-1,051,266 

#     ax.set_title(title, y =0.8)
#     sns.despine(top=True, right=True, bottom=True)
#   ax.set_xlabel('chr18:936,578-1,051,266')

# #   plt.gcf().get_axes()[0].set_ylim(0, 3)
# #   plt.gcf().get_axes()[1].set_ylim(0, 3)
# #   plt.gcf().get_axes()[2].set_ylim(0, 70)
# #   plt.gcf().get_axes()[3].set_ylim(0, 3)
#   plt.tight_layout()
# #   plt.savefig(f'Human_ATAC_alltracks.png')
#   plt.savefig(figname)

# output_seq = torch.load(f'/exports/humgen/idenhond/data/Enformer_train/Enformer_train_output_humanatac/output_seq1.pt', map_location=torch.device('cpu')).squeeze()
# target_tensor = torch.load(f'/exports/humgen/idenhond/data/Enformer_train/Human_ATAC_train_targets/targets_seq1.pt', map_location=torch.device('cpu')).squeeze()
# tracks = {'Human ATAC GABAergic tfr': target[:, 22],
#           'Human ATAC GABAergic target tensor': target_tensor[:, 22],
#           'Human ATAC GABAergic prediction': output_seq[:, 22],
#           'Human ATAC Oligo tfr': target[:, 55],
#           'Human ATAC Oligo target tensor': target_tensor[:, 55],
#           'Human ATAC Oligo prediction': output_seq[:, 55]}
# figname = 'test-plot_atac.png'

# plot_tracks(tracks, None, figname)


'''
28/4: plot tracks like in paper, test sequence 1
track 22 = GABAergic = test correlation  0.726330
track 24 = Glutamatergic = test correlation = 0.719774 
'''
# def plot_tracks(tracks, interval, track_nr, height=1.5):
#   fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
#   for i, (ax, (title, y)) in enumerate(zip(axes, tracks.items())):
#     print(i, title) 
#     if i == 0 or i == 1:
#         color = 'steelblue'
#     else:
#         color = 'sandybrown'
  
#     ax.fill_between(np.linspace(0, 114688, num=len(y)), y, color = color) 

#     ax.set_title(title, y = 0.8)
#     sns.despine(top=True, right=True, bottom=True)
#   ax.set_xlabel(f'test seq {track_nr}') # test seq 1

#   plt.tight_layout()
#   plt.savefig(f'Human_ATAC_track22_GABA_track24_GLUTA_testseq{track_nr}.png')
#   plt.close()
#   # plt.savefig(figname)

# for i in range(1, 30):
#   output_seq = torch.load(f'/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output_humanatac/output_seq{i}.pt', map_location=torch.device('cpu')).squeeze()
#   target_tensor = torch.load(f'/exports/humgen/idenhond/data/Enformer_test/Human_ATAC_test_targets/targets_seq{i}.pt', map_location=torch.device('cpu')).squeeze()
#   print(output_seq.shape)
#   print(target_tensor.shape)
#   tracks = {'Human ATAC GABAergic target': target_tensor[:, 22],
#             'Human ATAC GABAergic prediction': output_seq[:, 22],
#             'Human ATAC Glutamatergic target': target_tensor[:, 24],
#             'Human ATAC Glutamatergic prediction': output_seq[:, 24]}

#   # tracks = {'Human ATAC GABAergic target': target_tensor[:, 22],
#   #           'Human ATAC GABAergic prediction': output_seq[:, 22]}

#   plot_tracks(tracks, None, i)

def plot_tracks(tracks, interval, height=2):
  with sns.plotting_context("talk"):
    fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
    fig_width, fig_height = plt.gcf().get_size_inches()
    print(fig_width, fig_height)
    for i, (ax, (title, y)) in enumerate(zip(axes, tracks.items())):
      print(i, title) 
      if i == 0 or i == 1:
          color = 'steelblue'
      else:
          color = 'sandybrown'
    
      ax.fill_between(np.linspace(10590327, 10705015, num=len(y)), y, color = color) 

      ax.set_title(title, y = 1)
      sns.despine(top=True, right=True, bottom=True)
    ax.set_xlabel(f'chr19:10590327-10705015') # test seq 1

    plt.tight_layout()
    plt.savefig(f'Human_ATAC_track22_GABA_track55_OLIGO_testseq26_talk.png')
    plt.close()
  # plt.savefig(figname)

# for i in range(1, 30):
#   output_seq = torch.load(f'/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output_humanatac/output_seq{i}.pt', map_location=torch.device('cpu')).squeeze()
#   target_tensor = torch.load(f'/exports/humgen/idenhond/data/Enformer_test/Human_ATAC_test_targets/targets_seq{i}.pt', map_location=torch.device('cpu')).squeeze()
#   print(output_seq.shape)
#   print(target_tensor.shape)
#   tracks = {'Human ATAC GABAergic target': target_tensor[:, 22],
#             'Human ATAC GABAergic prediction': output_seq[:, 22],
#             'Human ATAC Oligo target': target_tensor[:, 55],
#             'Human ATAC Oligo prediction': output_seq[:, 55]}

#   # tracks = {'Human ATAC GABAergic target': target_tensor[:, 22],
#   #           'Human ATAC GABAergic prediction': output_seq[:, 22]}

#   plot_tracks(tracks, None, i)

output_seq = torch.load(f'/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output_humanatac/output_seq26.pt', map_location=torch.device('cpu')).squeeze()
target_tensor = torch.load(f'/exports/humgen/idenhond/data/Enformer_test/Human_ATAC_test_targets/targets_seq26.pt', map_location=torch.device('cpu')).squeeze()
print(output_seq.shape)
print(target_tensor.shape)
tracks = {'Human ATAC GABAergic target': target_tensor[:, 22],
          'Human ATAC GABAergic prediction': output_seq[:, 22],
          'Human ATAC Oligo target': target_tensor[:, 55],
          'Human ATAC Oligo prediction': output_seq[:, 55]}
plot_tracks(tracks, None)

