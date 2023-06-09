import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
Compare performance of 27 new tracks to new-tracks-model
"""

input_file = '/exports/humgen/idenhond/data/basenji_preprocess/output_tfr_27newtracks_human/targets.txt'
df = pd.read_csv(input_file, sep = '\t')
print(df)
print(f"Number of trakcs per assay type: \n {df['assay type'].value_counts()}")

# read csv with correlation score per track for test and validation sequences
col_name = ['correlation test all tracks model']
df_correlation_test_alltracks = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_alltracks_2404.csv', index_col = 0, names = col_name).tail(-1).tail(27).reset_index()
col_name = ['correlation test new tracks model']
df_correlation_test_newtracks = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_newtracks_2404.csv', index_col = 0, names = col_name).tail(-1)
col_name = ['correlation validation all tracks model']
df_correlation_validation_alltracks = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_alltracks_2404.csv', index_col = 0, names = col_name).tail(-1).tail(27).reset_index()
col_name = ['correlation validation new tracks model']
df_correlation_validation_newtracks = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_newtracks_2404.csv', index_col = 0, names = col_name).tail(-1)

col_name = ['correlation train all tracks model']
df_correlation_train_alltracks = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_alltracks_2404.csv', index_col = 0, names = col_name).tail(-1).tail(27).reset_index()
col_name = ['correlation train new tracks model']
df_correlation_train_newtracks = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_newtracks_2404.csv', index_col = 0, names = col_name).tail(-1)

# print(df_correlation_validation_alltracks.shape)
# print(df_correlation_validation_newtracks.shape)

# print(df_correlation_test_newtracks)
# print(df_correlation_test_alltracks)

df['correlation test all tracks model'] = df_correlation_test_alltracks['correlation test all tracks model']
df['correlation test new tracks model'] = df_correlation_test_newtracks['correlation test new tracks model']
df['correlation validation all tracks model'] = df_correlation_validation_alltracks['correlation validation all tracks model']
df['correlation validation new tracks model'] = df_correlation_validation_newtracks['correlation validation new tracks model']
df['correlation train all tracks model'] = df_correlation_train_alltracks['correlation train all tracks model']
df['correlation train new tracks model'] = df_correlation_train_newtracks['correlation train new tracks model']

print(df[['index', 'identifier', 'correlation test all tracks model']])

print('test')
print(df['correlation test all tracks model'].min())
print(df['correlation test all tracks model'].max())
print('validation')
print(df['correlation validation all tracks model'].min())
print(df['correlation validation all tracks model'].max())
print('train')
print(df['correlation train all tracks model'].min())
print(df['correlation train all tracks model'].max())

a = df['correlation test all tracks model'].mean()
print(f'test correlation: {a}\n')
a = df['correlation test new tracks model'].mean()
print(f'test correlation dnn head: {a}\n')
a = df['correlation train all tracks model'].mean()
print(f'train correlation: {a}\n')
a = df['correlation train new tracks model'].mean()
print(f'train correlation dnn head: {a}\n')
a = df['correlation validation all tracks model'].mean()
print(f'validation correlation: {a}\n')
a = df['correlation validation new tracks model'].mean()
print(f'validation correlation dnn head: {a}\n')

plt.figure(figsize = (4.8, 4.8))
legend_map = {'Histone ChIP': 'Histone ChIP',
              'TF ChIP': 'TF ChIP',
              'DNASE': 'DNase'}
plt.axline((0.2, 0.2), (0.85, 0.85), linewidth=0.5, color='k', linestyle = 'dashed')
ax = sns.scatterplot(data = df, s = 100, x = 'correlation test all tracks model', y = 'correlation test all tracks model', hue = df['assay type'].map(legend_map), palette = sns.color_palette("Paired", 3))
plt.xlabel('Human head model trained on new tracks')
plt.ylabel('Human head model trained on all tracks')
plt.legend(title = None)
ax.text(1, 0.03, '0.658', fontsize = 9, ha='right', va='center', transform=ax.transAxes)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1]) 
ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1]) 
ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
sns.despine(top=True, right=True, left=False, bottom=False)
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig2_newtracks/Scatter_corr_test_newtracks.png', bbox_inches='tight', dpi = 300)
plt.close()
fig_width, fig_height = plt.gcf().get_size_inches()
print(fig_width, fig_height)

plt.figure(figsize = (4.8, 4.8))
plt.axline((0.17, 0.17), (0.8, 0.8), linewidth=0.5, color='k', linestyle = 'dashed')
ax = sns.scatterplot(data = df, s = 100, x = 'correlation validation all tracks model', y = 'correlation validation all tracks model', hue = df['assay type'].map(legend_map), palette = sns.color_palette("Paired", 3))
plt.xlabel('Human head model trained on new tracks')
plt.ylabel('Human head model trained on all tracks')
plt.title('Validation set')
ax.text(1, 0.03, '0.628', fontsize = 9, ha='right', va='center', transform=ax.transAxes)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1]) 
ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1]) 
ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
plt.legend(title = None)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig2_newtracks/Scatter_corr_validation_newtracks.png', bbox_inches='tight', dpi = 300)
plt.close()

plt.figure(figsize = (4.8, 4.8))
plt.axline((0.17, 0.17), (0.8, 0.8), linewidth=0.5, color='k', linestyle = 'dashed')
ax = sns.scatterplot(data = df, s = 100, x = 'correlation train all tracks model', y = 'correlation train all tracks model', hue = df['assay type'].map(legend_map), palette = sns.color_palette("Paired", 3))
plt.xlabel('Human head model trained on new tracks')
plt.ylabel('Human head model trained on all tracks')
plt.title('Train set')
plt.legend(title = None)
ax.text(1, 0.03, '0.628', fontsize = 9, ha='right', va='center', transform=ax.transAxes)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1]) 
ax.set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1]) 
ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1])
sns.despine(top=True, right=True, left=False, bottom=False)
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig2_newtracks/Scatter_corr_train_newtracks.png', bbox_inches='tight', dpi = 300)
plt.close()


exit()
"""
Compare performance of 5313  tracks to dnn head model
"""

input_file = '/exports/humgen/idenhond/data/Basenji/human-targets.txt'
df = pd.read_csv(input_file, sep = '\t')
df[['assay type', 'description2']] = df.description.str.split(':', n = 1, expand = True) # make new column for assay type
def f(row):
    if row['assay type'] == 'CHIP':
        if any(row['description2'].startswith(x) for x in ['H2AK', 'H2BK', 'H3K', 'H4K']): val = 'ChIP Histone'
        else: val = 'ChIP TF'
    elif row['assay type'] == 'DNASE' or row['assay type'] == 'ATAC': val = 'DNASE/ATAC'
    else: val = row['assay type']
    return val
df['assay type split ChIP'] = df.apply(f, axis=1)
print(f'Number of tracks: {df.shape[0]}\n')
print(f"Number of trakcs per assay type: \n {df['assay type split ChIP'].value_counts()}\n")

# read csv with correlation score per track for test and validation sequences
col_name = ['correlation test']
df_correlation_test = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_alltracks_2404.csv', index_col = 0, names = col_name)
print(f'df correlation test all tracks shape: {df_correlation_test.shape}')
col_name = ['correlation test enformer']
df_correlation_test_enformer = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_own_output_pretrainedmodel.csv', index_col = 0, names = col_name)
col_name = ['correlation test dnn head']
df_correlation_test_dnnhead = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_dnn_head.csv', index_col = 0, names = col_name)
print(f'df correlation dnn head shape: {df_correlation_test_dnnhead.shape}')

col_name = ['correlation validation']
df_correlation_validation = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_alltracks_2404.csv', index_col = 0, names = col_name)
col_name = ['correlation valid enformer']
df_correlation_validation_enformer = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_own_output_pretrainedmodel.csv', index_col = 0, names = col_name)
col_name = ['correlation validation dnn head']
df_correlation_validation_dnnhead = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_dnn_head.csv', index_col = 0, names = col_name)

col_name = ['correlation train']
df_correlation_train = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train_dnn_head.csv', index_col = 0, names = col_name)
col_name = ['correlation train enformer']
df_correlation_train_enformer = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train.csv', index_col = 0, names = col_name)
col_name = ['correlation train dnn head']
df_correlation_train_dnnhead = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train_dnn_head.csv', index_col = 0, names = col_name)

df_correlation_test = df_correlation_test.tail(-1)
df_correlation_test_enformer = df_correlation_test_enformer.tail(-1)
df_correlation_test_dnnhead = df_correlation_test_dnnhead.tail(-1)
df_correlation_validation = df_correlation_validation.tail(-1)
df_correlation_validation_enformer = df_correlation_validation_enformer.tail(-1)
df_correlation_validation_dnnhead = df_correlation_validation_dnnhead.tail(-1)
df_correlation_train = df_correlation_train.tail(-1)
df_correlation_train_enformer = df_correlation_train_enformer.tail(-1)
df_correlation_train_dnnhead = df_correlation_train_dnnhead.tail(-1)

# add column to df with test and validation correlation scores
df['test correlation'] = df_correlation_test['correlation test']
df['test correlation dnn head'] = df_correlation_test_dnnhead['correlation test dnn head']
df['test correlation enformer'] = df_correlation_test_enformer['correlation test enformer']
df['validation correlation'] = df_correlation_validation['correlation validation']
df['validation correlation dnn head'] = df_correlation_validation_dnnhead['correlation validation dnn head']
df['validation correlation enformer'] = df_correlation_validation_enformer['correlation valid enformer']
df['train correlation'] = df_correlation_train['correlation train']
df['train correlation dnn head'] = df_correlation_train_dnnhead['correlation train dnn head']
df['train correlation enformer'] = df_correlation_train_enformer['correlation train enformer']

# calculate mean test and validation correlation score
print(f'mean correlation score test all tracks model: {df["test correlation"].mean(axis=0):.4f}')
print(f'mean correlation score test enformer: {df["test correlation enformer"].mean(axis=0):.4f}')
print(f'mean correlation score test dnn head: {df["test correlation dnn head"].mean(axis=0):.4f}')
print(f'mean correlation score validation all tracks model: {df["validation correlation"].mean(axis=0):.4f}')
print(f'mean correlation score validation enformer: {df["validation correlation enformer"].mean(axis=0):.4f}')
print(f'mean correlation score validation dnn head: {df["validation correlation dnn head"].mean(axis=0):.4f}')
print(f'mean correlation score train all tracks model: {df["train correlation"].mean(axis=0):.4f}')
print(f'mean correlation score train enformer: {df["train correlation enformer"].mean(axis=0):.4f}')
print(f'mean correlation score train dnn head: {df["train correlation dnn head"].mean(axis=0):.4f}')

a = df.groupby('assay type split ChIP')['test correlation'].mean()
print(f'test correlation: {a}\n')
a = df.groupby('assay type split ChIP')['test correlation dnn head'].mean()
print(f'test correlation dnn head: {a}\n')
a = df.groupby('assay type split ChIP')['train correlation'].mean()
print(f'train correlation: {a}\n')
a = df.groupby('assay type split ChIP')['train correlation dnn head'].mean()
print(f'train correlation dnn head: {a}\n')
a = df.groupby('assay type split ChIP')['validation correlation'].mean()
print(f'validation correlation: {a}\n')
a = df.groupby('assay type split ChIP')['validation correlation dnn head'].mean()
print(f'validation correlation dnn head: {a}\n')

sns.plotting_context("paper")
fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, constrained_layout=True, figsize=(6, 3.8))
for i, (key, value) in enumerate(df['assay type split ChIP'].value_counts().to_dict().items()):
    print(i, key)
    ax[i].set(adjustable='box', aspect='equal')
    df_subset = df[df['assay type split ChIP'] == key]
    ax[i].axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
    sns.scatterplot(data = df_subset, s = 20, x = 'train correlation', y = 'train correlation dnn head', color = 'k', ax = ax[i], edgecolor=['white'])
    ax[i].set(xlabel=None)
    ax[i].set(ylabel=None)
    if key == 'DNASE/ATAC': ax[i].set_title(f'DNase, ATAC', fontsize = 5)
    else: ax[i].set_title(f'{key}', fontsize = 5)
    ax[i].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1]) 
    ax[i].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=5)
    ax[i].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1]) 
    ax[i].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=5)
    ax[i].spines[['right', 'top']].set_visible(False)
    # chip tf
    ax[0].text(0.1, 0.95, '0.553', fontsize = 5, ha='center', va='center', transform=ax[0].transAxes)
    ax[0].text(0.9, 0.03, '0.553', fontsize = 5, ha='center', va='center', transform=ax[0].transAxes)
    # chip histone. 
    ax[1].text(0.1, 0.95, '0.698', fontsize = 5, ha='center', va='center', transform=ax[1].transAxes)
    ax[1].text(0.9, 0.03, '0.699', fontsize = 5, ha='center', va='center', transform=ax[1].transAxes)
    # dnase atac.
    ax[2].text(0.1, 0.95, '0.720', fontsize = 5, ha='center', va='center', transform=ax[2].transAxes)
    ax[2].text(0.9, 0.03, '0.720', fontsize = 5, ha='center', va='center', transform=ax[2].transAxes)
    # cage. 
    ax[3].text(0.1, 0.95, '0.682', fontsize = 5, ha='center', va='center', transform=ax[3].transAxes)
    ax[3].text(0.9, 0.03, '0.682', fontsize = 5, ha='center', va='center', transform=ax[3].transAxes)
plt.figtext(.5, .25, f'Human head model trained on all tracks', fontsize = 6, ha='center')
fig.supylabel(f'      Human head model \ntrained on Enformer tracks', fontsize = 6)
plt.figtext(.5, .75, f'Train set', fontsize = 8, ha='center')
fig.tight_layout()
plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig2_newtracks/Scatter_corr_train_alltracks.png', bbox_inches='tight', dpi = 300)

sns.plotting_context("paper")
fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, constrained_layout=True, figsize=(6, 3.8))
for i, (key, value) in enumerate(df['assay type split ChIP'].value_counts().to_dict().items()):
    print(i, key)
    ax[i].set(adjustable='box', aspect='equal')
    df_subset = df[df['assay type split ChIP'] == key]
    ax[i].axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
    sns.scatterplot(data = df_subset, s = 20, x = 'validation correlation', y = 'validation correlation dnn head', color = 'k', ax = ax[i], edgecolor=['white'])
    ax[i].set(xlabel=None)
    ax[i].set(ylabel=None)
    if key == 'DNASE/ATAC': ax[i].set_title(f'DNase, ATAC', fontsize = 5)
    else: ax[i].set_title(f'{key}', fontsize = 5)
    ax[i].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1]) 
    ax[i].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=5)
    ax[i].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1]) 
    ax[i].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=5)
    ax[i].spines[['right', 'top']].set_visible(False)
    # chip tf
    ax[0].text(0.1, 0.95, '0.482', fontsize = 5, ha='center', va='center', transform=ax[0].transAxes)
    ax[0].text(0.9, 0.03, '0.482', fontsize = 5, ha='center', va='center', transform=ax[0].transAxes)
    # chip histone. 
    ax[1].text(0.1, 0.95, '0.618', fontsize = 5, ha='center', va='center', transform=ax[1].transAxes)
    ax[1].text(0.9, 0.03, '0.618', fontsize = 5, ha='center', va='center', transform=ax[1].transAxes)
    # dnase atac.
    ax[2].text(0.1, 0.95, '0.609', fontsize = 5, ha='center', va='center', transform=ax[2].transAxes)
    ax[2].text(0.9, 0.03, '0.609', fontsize = 5, ha='center', va='center', transform=ax[2].transAxes)
    # cage. 
    ax[3].text(0.1, 0.95, '0.588', fontsize = 5, ha='center', va='center', transform=ax[3].transAxes)
    ax[3].text(0.9, 0.03, '0.588', fontsize = 5, ha='center', va='center', transform=ax[3].transAxes)
plt.figtext(.5, .25, f'Human head model trained on all tracks', fontsize = 6, ha='center')
fig.supylabel(f'      Human head model \ntrained on Enformer tracks', fontsize = 6)
plt.figtext(.5, .75, f'Validation set', fontsize = 8, ha='center')
fig.tight_layout()
plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig2_newtracks/Scatter_corr_validation_alltracks.png', bbox_inches='tight', dpi = 300)

sns.plotting_context("paper")
fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, constrained_layout=True, figsize=(6, 3.8))
for i, (key, value) in enumerate(df['assay type split ChIP'].value_counts().to_dict().items()):
    print(i, key)
    ax[i].set(adjustable='box', aspect='equal')
    df_subset = df[df['assay type split ChIP'] == key]
    ax[i].axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
    sns.scatterplot(data = df_subset, s = 20, x = 'test correlation', y = 'test correlation dnn head', color = 'k', ax = ax[i], edgecolor=['white'])
    ax[i].set(xlabel=None)
    ax[i].set(ylabel=None)
    if key == 'DNASE/ATAC': ax[i].set_title(f'DNase, ATAC', fontsize = 5)
    else: ax[i].set_title(f'{key}', fontsize = 5)
    ax[i].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1]) 
    ax[i].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=5)
    ax[i].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1]) 
    ax[i].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=5)
    ax[i].spines[['right', 'top']].set_visible(False)
    # chip tf
    ax[0].text(0.1, 0.95, '0.504', fontsize = 5, ha='center', va='center', transform=ax[0].transAxes)
    ax[0].text(0.9, 0.03, '0.504', fontsize = 5, ha='center', va='center', transform=ax[0].transAxes)
    # chip histone. 
    ax[1].text(0.1, 0.95, '0.642', fontsize = 5, ha='center', va='center', transform=ax[1].transAxes)
    ax[1].text(0.9, 0.03, '0.642', fontsize = 5, ha='center', va='center', transform=ax[1].transAxes)
    # dnase atac.
    ax[2].text(0.1, 0.95, '0.698', fontsize = 5, ha='center', va='center', transform=ax[2].transAxes)
    ax[2].text(0.9, 0.03, '0.698', fontsize = 5, ha='center', va='center', transform=ax[2].transAxes)
    # cage. 
    ax[3].text(0.1, 0.95, '0.575', fontsize = 5, ha='center', va='center', transform=ax[3].transAxes)
    ax[3].text(0.9, 0.03, '0.575', fontsize = 5, ha='center', va='center', transform=ax[3].transAxes)
plt.figtext(.5, .25, f'Human head model trained on all tracks', fontsize = 6, ha='center')
fig.supylabel(f'      Human head model \ntrained on Enformer tracks', fontsize = 6)
plt.figtext(.5, .75, f'Test set', fontsize = 8, ha='center')
fig.tight_layout()
plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig2_newtracks/Scatter_corr_test_alltracks.png', bbox_inches='tight', dpi = 300)
