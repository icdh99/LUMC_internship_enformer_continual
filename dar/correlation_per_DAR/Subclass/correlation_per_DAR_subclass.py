import pandas as pd
import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import seaborn as sns

df_enformer = pd.read_csv('/exports/humgen/idenhond/data/DAR/Enformer_test/enformer_test.bed', sep = '\t', names = ['chr', 'start', 'stop'])

df = pd.read_csv('/exports/humgen/idenhond/data/DAR/Subclass_level/intersect_14E_test_wb_sorted.bed', sep = '\t', names = ['chr', 'start', 'stop', 'chr2', 'start2', 'stop2', 'Subclass', 'FullName'], index_col=False)
print(f'We start with {len(df)} sequences from the intersect between table 14E and the Enformer test set')
df['length'] = df['stop'] - df['start']
df = df[df['length'] >= 256].reset_index()
print(f'After keeping only the sequences that are equal to or longer than 256 bp we keep {len(df)} sequences in the intersect\n')
print(df)

sequences = 0
skipped_sequences = 0
output_list = []
target_list = []
nr_seqs = []

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
    nr_seqs.append(i)
    assert nr_bins >= 2, f"minimum of 2 bins expected, got: {nr_bins}"
    assert stop_bin > start_bin, f'stop bin should be bigger than start bin, got: start {start_bin} and stop {stop_bin}'
    # index of enformer test sequence
    index_enformer = subset_df.index[0]
    # load torch 
    atac_output = torch.load(f'/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output_humanatac/output_seq{index_enformer+1}.pt', map_location=torch.device('cpu')).squeeze()
    atac_target = torch.load(f'/exports/humgen/idenhond/data/Enformer_test/Human_ATAC_test_targets/targets_seq{index_enformer+1}.pt', map_location=torch.device('cpu')).squeeze()
    # get only subclass tracks without duplicates
    idx_subclass = [51, 60, 61, 62, 63, 64, 43, 45, 46, 44, 48, 49, 50, 47] # 14 neuronal subclasses
    # get correct bins
    output = atac_output[start_bin:stop_bin+1, idx_subclass].numpy()
    target = atac_target[start_bin:stop_bin+1, idx_subclass].numpy()
    means = np.mean(output, axis = 0)
    means_target = np.mean(target, axis = 0)
    output_list.append(list(means))
    target_list.append(list(means_target))

    if i == 10: break

print(f'There are {skipped_sequences} sequences skipped')
print(len(output_list))
print(len(nr_seqs))
print(max(nr_seqs))

a = np.array(output_list).transpose()
a_target = np.array(target_list).transpose()
print(a.shape, a_target.shape)

filepath = 'Predictions_test_DAR_subclass.csv'
np.savetxt(filepath, a, delimiter=",", fmt='%.3e')
filepath = 'Targets_test_DAR_subclass.csv'
np.savetxt(filepath, a_target, delimiter=",", fmt ='%.3e')

# 1x gerund
original_seq_nr_list = []
corr_list = []
full_name_list = []
for i, nr in enumerate(((nr_seqs))):
    corr = np.corrcoef(a[:, i], a_target[:, i])[0, 1]
    # print(i, nr, corr)
    original_seq_nr_list.append(nr)
    corr_list.append(corr)
    fullname = df["Subclass"].iloc[nr]
    full_name_list.append(fullname)

print(len(original_seq_nr_list))
print(len(corr_list))
print(len(full_name_list))

df = pd.DataFrame({'Original Seq nr': original_seq_nr_list, 'Correlation': corr_list, 'Full name': full_name_list})
print(df)

df.to_csv('/exports/humgen/idenhond/projects/dar/correlation_per_DAR/Subclass/Correlation_DAR_subclass.csv')

