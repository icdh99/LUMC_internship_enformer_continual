import pandas as pd
import numpy as np
import math
import torch
import matplotlib.pyplot as plt
import seaborn as sns

df_subclasses = pd.read_csv('/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_Subclass.csv', sep = '\t')
print(df_subclasses)
subclass_labels = df_subclasses['Subclass'].to_list()[:14]
print(subclass_labels)
print(len(subclass_labels))

df_enformer = pd.read_csv('/exports/humgen/idenhond/data/DAR/Enformer_test/enformer_test.bed', sep = '\t', names = ['chr', 'start', 'stop'])
print(df_enformer)

df = pd.read_csv('/exports/humgen/idenhond/data/DAR/Subclass_level/intersect_14E_test.bed', sep = '\t', names = ['chr', 'start', 'stop'])
print(f'We start with {len(df)} sequences from the intersect between table 14E and the Enformer test set')
df['length'] = df['stop'] - df['start']
df = df[df['length'] >= 256]
print(f'After keeping only the sequences that are equal to or longer than 256 bp we keep {len(df)} sequences in the intersect\n')

# print(f'The first intersect sequence:')
# row = df.iloc[[0]]
# print(row)
# start_intersect = row['start'][0] # this is a Series object
# stop_intersect = row['stop'][0] # this is a Series object
# chrom = row['chr'][0]

# print(start_intersect)
# print(stop_intersect)
# print(chrom)

# indices = np.where((df_enformer['chr'] == chrom) & (df_enformer['start'] <= start_intersect) & (df_enformer['stop'] >= stop_intersect))[0]
# print(indices)
# subset_df = df_enformer.iloc[indices]
# print(subset_df)

skipped_sequences = 0
output_list = []
target_list = []

for i, row in enumerate(df.itertuples()):
    start_intersect = row.start # this is a Series object
    stop_intersect = row.stop # this is a Series object
    chrom = row.chr
    # print(i, start_intersect, stop_intersect, chrom)

    indices = np.where((df_enformer['chr'] == chrom) & (df_enformer['start'] <= start_intersect) & (df_enformer['stop'] >= stop_intersect))[0]
    # print(indices)
    subset_df = df_enformer.iloc[indices]
    # print(subset_df)
    start_enformer = int(subset_df['start'])
    # print(start_enformer)

    start_bin = math.ceil((start_intersect - start_enformer - 1 )/128 + 1)
    assert start_bin > 0 and start_bin < 896, f"startbin should be in between 0 and 896, got: {start_bin}"
    # print(start_bin)
    stop_bin = math.floor((stop_intersect - start_enformer) / 128)
    # print(stop_bin)
    assert stop_bin > 0 and stop_bin <= 896, f"stopbin should be in between 0 and 896, got: {stop_bin}"
    nr_bins = stop_bin - start_bin + 1
    if nr_bins < 2: 
        skipped_sequences += 1
        continue
    # print(nr_bins)
    assert nr_bins >= 2, f"minimum of 2 bins expected, got: {nr_bins}"
    assert stop_bin > start_bin, f'stop bin should be bigger than start bin, got: start {start_bin} and stop {stop_bin}'

    # index of enformer test sequence
    index_enformer = subset_df.index[0]
    # print(index_enformer)

    # load torch 
    atac_output = torch.load(f'/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output_humanatac/output_seq{index_enformer+1}.pt', map_location=torch.device('cpu')).squeeze()
    atac_target = torch.load(f'/exports/humgen/idenhond/data/Enformer_test/Human_ATAC_test_targets/targets_seq{index_enformer+1}.pt', map_location=torch.device('cpu')).squeeze()

    # print(atac_output.shape)
    # get only subclass tracks
    idx_subclass = [51, 60, 61, 62, 63, 64, 43, 45, 46, 44, 48, 49, 50, 47]
    # get correct bins
    output = atac_output[start_bin:stop_bin+1, idx_subclass].numpy()
    target = atac_target[start_bin:stop_bin+1, idx_subclass].numpy()
    # print(output.shape)
    means = np.mean(output, axis = 0)
    means_target = np.mean(target, axis = 0)
    # print(means)
    # print(means.shape)
    # print(list(means))
    output_list.append(list(means))
    target_list.append(list(means_target))


    # print('\n')
    # if i == 5: break


print(f'There are {skipped_sequences} sequence skipped')

# print(output_list)
print(len(output_list))
a = np.array(output_list).transpose()
a_target = np.array(target_list).transpose()
print(a)
print(a.shape)

row_height = 45
column_width = 0.007
num_rows, num_columns = a.shape
fig_width = num_columns * column_width
fig_height = num_rows * row_height

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.set_aspect(column_width / row_height)
plt.imshow(a, interpolation='none', cmap='viridis', origin = 'upper', extent = [0, 2514, 0, 14], aspect=180) 
ax.set_yticks(np.arange(len(subclass_labels)) + 0.5, subclass_labels)
ax.set_yticklabels(subclass_labels, va='center')
plt.colorbar(shrink=0.01)
plt.savefig('Plot_Intersect_PredictedValues.png', bbox_inches='tight', dpi = 100)
plt.savefig('Plot_Intersect_PredictedValues.pdf', bbox_inches='tight', dpi = 100)
fig_width, fig_height = plt.gcf().get_size_inches()
print(fig_width, fig_height)

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.set_aspect(column_width / row_height)
plt.imshow(a_target, interpolation='none', cmap='viridis', origin = 'upper', extent = [0, 2514, 0, 14], aspect=180) # 2514 / 14 en dan afronden
ax.set_yticks(np.arange(len(subclass_labels)) + 0.5, subclass_labels)
ax.set_yticklabels(subclass_labels, va='center')
plt.colorbar(shrink=0.01)
plt.savefig('Plot_Intersect_TargetValues.png', bbox_inches='tight', dpi = 100)
plt.savefig('Plot_Intersect_TargetValues.pdf', bbox_inches='tight', dpi = 100)

a_target_clipped = np.clip(a_target, 0, 5)
fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.set_aspect(column_width / row_height)
plt.imshow(a_target_clipped, interpolation='none', cmap='viridis', origin = 'upper', extent = [0, 2514, 0, 14], aspect=180)
ax.set_yticks(np.arange(len(subclass_labels)) + 0.5, subclass_labels)
ax.set_yticklabels(subclass_labels, va='center')
plt.colorbar(shrink=0.01)
plt.savefig('Plot_Intersect_TargetValues_Clipped.png', bbox_inches='tight', dpi = 100)
plt.savefig('Plot_Intersect_TargetValues_Clipped.pdf', bbox_inches='tight', dpi = 100)

def normalize(x):
    x = np.asarray(x)
    print(x.shape)
    return (x - x.min()) / (np.ptp(x))


fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.set_aspect(column_width / row_height)
plt.imshow(normalize(a), interpolation='none', cmap='viridis', origin = 'upper', extent = [0, 2514, 0, 14], aspect=180) 
ax.set_yticks(np.arange(len(subclass_labels)) + 0.5, subclass_labels)
ax.set_yticklabels(subclass_labels, va='center')
plt.colorbar(shrink=0.01)
plt.savefig('Plot_Intersect_PredictedValues_Normalized.png', bbox_inches='tight', dpi = 100)
plt.savefig('Plot_Intersect_PredictedValues_Normalized.pdf', bbox_inches='tight', dpi = 100)

fig, ax = plt.subplots(figsize=(fig_width, fig_height))
ax.set_aspect(column_width / row_height)
plt.imshow(normalize(a_target), interpolation='none', cmap='viridis', origin = 'upper', extent = [0, 2514, 0, 14], aspect=180)
ax.set_yticks(np.arange(len(subclass_labels)) + 0.5, subclass_labels)
ax.set_yticklabels(subclass_labels, va='center')
plt.colorbar(shrink=0.01)
plt.savefig('Plot_Intersect_TargetValues_Normalized.png', bbox_inches='tight', dpi = 100)
plt.savefig('Plot_Intersect_TargetValues_Normalized.pdf', bbox_inches='tight', dpi = 100)



# origin upper --> linksboven is 0.0 van je array
# extent = left right bottom top coordinates (nu de grenzen van je data = zelfde als default)
# aspect = default 1
# aspect auto --> kwart slag gedraaid, 14 rijen heel groot en hele smalle kolommen