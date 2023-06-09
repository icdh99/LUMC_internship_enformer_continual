import pandas as pd
import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import seaborn as sns

df_enformer_test = pd.read_csv('/exports/humgen/idenhond/data/DAR/Enformer_test/enformer_test.bed', sep = '\t', names = ['chr', 'start', 'stop'])
df_enformer_valid = pd.read_csv('/exports/humgen/idenhond/data/DAR/Enformer_test/enformer_valid.bed', sep = '\t', names = ['chr', 'start', 'stop'])
df_enformer_train = pd.read_csv('/exports/humgen/idenhond/data/DAR/Enformer_test/enformer_train.bed', sep = '\t', names = ['chr', 'start', 'stop'])

df = pd.read_csv('/exports/humgen/idenhond/data/DAR/AC_level/Cluster_14A.bed', sep = '\t', names = ['chr', 'start', 'stop', 'AC level', 'FullName'], index_col=False)
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

train, test, valid = 0, 0, 0

for i, row in enumerate(df.itertuples()):
    # if i == 5: break
    print(i)
    start_intersect = row.start # this is a Series object
    stop_intersect = row.stop # this is a Series object
    chrom = row.chr

    subset = None

    indices = np.where((df_enformer_test['chr'] == chrom) & (df_enformer_test['start'] <= start_intersect) & (df_enformer_test['stop'] >= stop_intersect))[0]
    if len(indices) == 1: 
        test += 1
        subset = 'test'
        subset_df = df_enformer_test.iloc[indices]
        # print(subset_df)
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
        index_enformer = subset_df.index[0]
        atac_output = torch.load(f'/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output_humanatac/output_seq{index_enformer+1}.pt', map_location=torch.device('cpu')).squeeze()
        atac_target = torch.load(f'/exports/humgen/idenhond/data/Enformer_test/Human_ATAC_test_targets/targets_seq{index_enformer+1}.pt', map_location=torch.device('cpu')).squeeze()
        idx_subclass = [35, 29,  30, 36, 37, 39, 40, 33, 25, 38, 41, 42, 31, 27, 28, 32, 8, 9, 10, 5, 6, 11, 12, 13, 19, 17, 15, 16, 21, 18, 20, 3, 1, 2, 52, 56, 57, 59] # zonder dubbele subclas clusters, 38 ac level clusters
        # get correct bins
        output = atac_output[start_bin:stop_bin+1, idx_subclass].numpy()
        target = atac_target[start_bin:stop_bin+1, idx_subclass].numpy()
        means = np.mean(output, axis = 0)
        means_target = np.mean(target, axis = 0)
        output_list.append(list(means))
        target_list.append(list(means_target))

    if len(indices) == 0:
        indices = np.where((df_enformer_valid['chr'] == chrom) & (df_enformer_valid['start'] <= start_intersect) & (df_enformer_valid['stop'] >= stop_intersect))[0]
        if len(indices) == 1: 
            valid += 1
            subset = 'valid'
            subset_df = df_enformer_valid.iloc[indices]
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
            index_enformer = subset_df.index[0]
            atac_output = torch.load(f'/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_output_humanatac/output_seq{index_enformer+1}.pt', map_location=torch.device('cpu')).squeeze()
            atac_target = torch.load(f'/exports/humgen/idenhond/data/Enformer_validation/Human_ATAC_validation_targets/targets_seq{index_enformer+1}.pt', map_location=torch.device('cpu')).squeeze()
            idx_subclass = [35, 29,  30, 36, 37, 39, 40, 33, 25, 38, 41, 42, 31, 27, 28, 32, 8, 9, 10, 5, 6, 11, 12, 13, 19, 17, 15, 16, 21, 18, 20, 3, 1, 2, 52, 56, 57, 59] # zonder dubbele subclas clusters, 38 ac level clusters
            # get correct bins
            output = atac_output[start_bin:stop_bin+1, idx_subclass].numpy()
            target = atac_target[start_bin:stop_bin+1, idx_subclass].numpy()
            means = np.mean(output, axis = 0)
            means_target = np.mean(target, axis = 0)
            output_list.append(list(means))
            target_list.append(list(means_target))

    if len(indices) == 0:
        # print(chrom, start_intersect, stop_intersect)
        indices = np.where((df_enformer_train['chr'] == chrom) & (df_enformer_train['start'] <= start_intersect) & (df_enformer_train['stop'] >= stop_intersect))[0]
        # print(indices)
        # print(len(indices))
        
        if len(indices) > 0: 
            subset = 'train'
            subset_df = df_enformer_train.iloc[indices]
            # print(subset_df)
            for row in subset_df.itertuples():
                train += 1
                start_enformer = row.start
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
                index_enformer = row.Index
                atac_output = torch.load(f'/exports/humgen/idenhond/data/Enformer_train/Enformer_train_output_humanatac/output_seq{index_enformer+1}.pt', map_location=torch.device('cpu')).squeeze()
                atac_target = torch.load(f'/exports/humgen/idenhond/data/Enformer_train/Human_ATAC_train_targets/targets_seq{index_enformer+1}.pt', map_location=torch.device('cpu')).squeeze()
                idx_subclass = [35, 29,  30, 36, 37, 39, 40, 33, 25, 38, 41, 42, 31, 27, 28, 32, 8, 9, 10, 5, 6, 11, 12, 13, 19, 17, 15, 16, 21, 18, 20, 3, 1, 2, 52, 56, 57, 59] # zonder dubbele subclas clusters, 38 ac level clusters
                # get correct bins
                output = atac_output[start_bin:stop_bin+1, idx_subclass].numpy()
                target = atac_target[start_bin:stop_bin+1, idx_subclass].numpy()
                means = np.mean(output, axis = 0)
                means_target = np.mean(target, axis = 0)
                output_list.append(list(means))
                target_list.append(list(means_target))

    if subset == None:
        skipped_sequences += 1
        continue


print(f'There are {skipped_sequences} sequences skipped')
print(len(output_list))
print(len(nr_seqs))
# print(max(nr_seqs))

print(f'train: {train}')
print(f'test: {test}')
print(f'valid: {valid}')
print(f'total: {train + test + valid}')

a = np.array(output_list).transpose()
a_target = np.array(target_list).transpose()
print(a.shape, a_target.shape)

filepath = 'Predictions_DAR_aclevel.csv'
np.savetxt(filepath, a, delimiter=",", fmt='%.3e')
filepath = 'Targets_DAR_aclevel.csv'
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
    fullname = df["FullName"].iloc[nr]
    full_name_list.append(fullname)

print(len(original_seq_nr_list))
print(len(corr_list))
print(len(full_name_list))

df = pd.DataFrame({'Original Seq nr': original_seq_nr_list, 'Correlation': corr_list, 'Full name': full_name_list})
print(df)

df.to_csv('/exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/Correlation_DAR_aclevel.csv')