import pandas as pd
import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import seaborn as sns

idx_subclass = [35, 29,  30, 36, 37, 39, 40, 33, 25, 38, 41, 42, 31, 27, 28, 32, 8, 9, 10, 5, 6, 11, 12, 13, 19, 17, 15, 16, 21, 18, 20, 3, 1, 2, 52, 56, 57, 59]
df_ac = pd.read_csv('/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_Ac-level_cluster.csv', sep = '\t').drop_duplicates(subset = 'Index old', keep = 'first', inplace=False)
df_ac = df_ac.loc[df_ac['Index old'].isin(idx_subclass)] #.sort_values(by = 'names')
idx_subclass_sorted = df_ac['Index old'].to_list()
print(idx_subclass_sorted)
print(df_ac['names'].to_list())
print(df_ac)

ac_labels = df_ac['names'].to_list()
# print(ac_labels)
print(len(ac_labels))

df_enformer = pd.read_csv('/exports/humgen/idenhond/data/DAR/Enformer_test/enformer_test.bed', sep = '\t', names = ['chr', 'start', 'stop'])
# print(df_enformer)

df = pd.read_csv('/exports/humgen/idenhond/data/DAR/AC_level/intersect_14A_test_withsubclasses_sorted.bed', sep = '\t', names = ['chr', 'start', 'stop', 'chr2', 'start2', 'stop2', 'AC level', 'FullName'], index_col=False)
print(f'We start with {len(df)} sequences from the intersect between table 14A and the Enformer test set')
df['length'] = df['stop'] - df['start']
df = df[df['length'] >= 256]
print(f'After keeping only the sequences that are equal to or longer than 256 bp we keep {len(df)} sequences in the intersect\n')
print(df)

sequences = 0
skipped_sequences = 0
output_list = []
target_list = []

for i, row in enumerate(df.itertuples()):
    start_intersect = row.start # this is a Series object
    stop_intersect = row.stop # this is a Series object
    chrom = row.chr
    indices = np.where((df_enformer['chr'] == chrom) & (df_enformer['start'] <= start_intersect) & (df_enformer['stop'] >= stop_intersect))[0]
    subset_df = df_enformer.iloc[indices]
    start_enformer = int(subset_df['start'])
    start_bin = math.ceil((start_intersect - start_enformer - 1 ) / 128 + 1)
    assert start_bin > 0 and start_bin < 896, f"startbin should be in between 0 and 896, got: {start_bin}"
    stop_bin = math.floor((stop_intersect - start_enformer) / 128)
    assert stop_bin > 0 and stop_bin <= 896, f"stopbin should be in between 0 and 896, got: {stop_bin}"
    nr_bins = stop_bin - start_bin + 1
    if nr_bins < 2: 
        skipped_sequences += 1
        continue
    assert nr_bins >= 2, f"minimum of 2 bins expected, got: {nr_bins}"
    assert stop_bin > start_bin, f'stop bin should be bigger than start bin, got: start {start_bin} and stop {stop_bin}'
    # index of enformer test sequence
    index_enformer = subset_df.index[0]
    # load torch 
    atac_output = torch.load(f'/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output_humanatac/output_seq{index_enformer+1}.pt', map_location=torch.device('cpu')).squeeze()
    atac_target = torch.load(f'/exports/humgen/idenhond/data/Enformer_test/Human_ATAC_test_targets/targets_seq{index_enformer+1}.pt', map_location=torch.device('cpu')).squeeze()
    # get only subclass tracks without duplicates
    idx_subclass = [35, 29,  30, 36, 37, 39, 40, 33, 25, 38, 41, 42, 31, 27, 28, 32, 8, 9, 10, 5, 6, 11, 12, 13, 19, 17, 15, 16, 21, 18, 20, 3, 1, 2, 52, 56, 57, 59]
    # get correct bins
    output = atac_output[start_bin:stop_bin+1, idx_subclass].numpy()
    target = atac_target[start_bin:stop_bin+1, idx_subclass].numpy()
    means = np.mean(output, axis = 0)
    means_target = np.mean(target, axis = 0)
    output_list.append(list(means))
    target_list.append(list(means_target))

print(f'There are {skipped_sequences} sequence skipped')
print(len(output_list))
a = np.array(output_list).transpose()
a_target = np.array(target_list).transpose()
print(a.shape, a_target.shape)

# labels

fig, ax = plt.subplots()
plt.imshow(a, interpolation='none', cmap='viridis', origin = 'lower', extent = [0, 16741, 0, 38], aspect=440)
ax.set_yticks(np.arange(len(ac_labels)) + 0.5, ac_labels)
ax.set_yticklabels(ac_labels, va='center', fontsize=4)
plt.colorbar(shrink=0.5)
plt.savefig('Plot_Intersect_PredictedValues.png', bbox_inches='tight', dpi = 200)

fig, ax = plt.subplots()
plt.imshow(a_target, interpolation='none', cmap='viridis', origin = 'lower', extent = [0, 16741, 0, 38], aspect=440)
plt.colorbar(shrink=0.5)
ax.set_yticks(np.arange(len(ac_labels)) + 0.5, ac_labels)
ax.set_yticklabels(ac_labels, va='center', fontsize=4)
plt.savefig('Plot_Intersect_TargetValues.png', bbox_inches='tight', dpi = 200)

def normalize(x):
    x = np.asarray(x)
    print(x.shape)
    return (x - x.min()) / (np.ptp(x))

fig, ax = plt.subplots()
plt.imshow(normalize(a_target), interpolation='none', cmap='viridis', origin = 'lower', extent = [0, 16741, 0, 38], aspect=440)
plt.colorbar(shrink=0.5)
# ax.set_yticks(np.arange(len(ac_labels)) + 0.5, ac_labels)
# ax.set_yticklabels(ac_labels, va='center', fontsize=4)
plt.savefig('Plot_Intersect_TargetValues_Normalized.png', bbox_inches='tight', dpi = 200)

fig, ax = plt.subplots()
plt.imshow(normalize(a), interpolation='none', cmap='viridis', origin = 'lower', extent = [0, 16741, 0, 38], aspect=440)
plt.colorbar(shrink=0.5)
# ax.set_yticks(np.arange(len(ac_labels)) + 0.5, ac_labels)
# ax.set_yticklabels(ac_labels, va='center', fontsize=4)
plt.savefig('Plot_Intersect_PredictedValues_Normalized.png', bbox_inches='tight', dpi = 200)






## clipped plots
# a = np.clip(a, 0, 1)
# fig, ax = plt.subplots()
# # ax.set_aspect(column_width / row_height)
# plt.imshow(a, interpolation='none', cmap='viridis', origin = 'upper', extent = [0, 16741, 0, 43], aspect=390)
# plt.colorbar(shrink=0.5)
# plt.savefig('Plot_Intersect_PredictedValues_Clipped.png', bbox_inches='tight', dpi = 200)

# a_target_clipped = np.clip(a_target, 0, 25)
# print(a_target_clipped)
# fig, ax = plt.subplots()
# # ax.set_aspect(column_width / row_height)
# plt.imshow(a_target_clipped, interpolation='none', cmap='viridis', origin = 'upper', extent = [0, 16741, 0, 43], aspect=390)
# plt.colorbar(shrink=0.5)
# plt.savefig('Plot_Intersect_TargetValues_Clipped.png', bbox_inches='tight', dpi = 200)