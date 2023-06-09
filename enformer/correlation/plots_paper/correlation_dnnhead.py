import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_file = '/exports/humgen/idenhond/data/Basenji/human-targets.txt'
df = pd.read_csv(input_file, sep = '\t')
df[['assay type', 'description2']] = df.description.str.split(':', n = 1, expand = True) # make new column for assay type

def f(row):
    if row['assay type'] == 'CHIP':
        if any(row['description2'].startswith(x) for x in ['H2AK', 'H2BK', 'H3K', 'H4K']): val = 'ChIP Histone'
        else: val = 'ChIP TF'
    elif row['assay type'] == 'DNASE' or row['assay type'] == 'ATAC': val = 'DNASE_ATAC'
    else: val = row['assay type']
    return val
df['assay type split ChIP'] = df.apply(f, axis=1)

print(f'Number of tracks: {df.shape[0]}\n')
print(f"Number of trakcs per assay type: \n {df['assay type'].value_counts()}")
print(f"Number of trakcs per assay type: \n {df['assay type split ChIP'].value_counts()}")

# read csv with correlation score per track for test and validation sequences
col_name = ['correlation test']
df_correlation_test = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_dnn_head.csv', index_col = 0, names = col_name)
col_name = ['correlation test enformer']
df_correlation_test_enformer = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_own_output_pretrainedmodel.csv', index_col = 0, names = col_name)
col_name = ['correlation validation']
df_correlation_validation = pd.read_csv('//exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_dnn_head.csv', index_col = 0, names = col_name)
col_name = ['correlation valid enformer']
df_correlation_validation_enformer = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_own_output_pretrainedmodel.csv', index_col = 0, names = col_name)
col_name = ['correlation train']
df_correlation_train = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train_dnn_head.csv', index_col = 0, names = col_name)
col_name = ['correlation train enformer']
df_correlation_train_enformer = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train.csv', index_col = 0, names = col_name)

df_correlation_test = df_correlation_test.tail(-1)
df_correlation_test_enformer = df_correlation_test_enformer.tail(-1)
df_correlation_validation = df_correlation_validation.tail(-1)
df_correlation_validation_enformer = df_correlation_validation_enformer.tail(-1)
df_correlation_train = df_correlation_train.tail(-1)
df_correlation_train_enformer = df_correlation_train_enformer.tail(-1)

# add column to df with test and validation correlation scores
df['test correlation'] = df_correlation_test['correlation test']
df['test correlation enformer'] = df_correlation_test_enformer['correlation test enformer']
df['validation correlation'] = df_correlation_validation['correlation validation']
df['validation correlation enformer'] = df_correlation_validation_enformer['correlation valid enformer']
df['train correlation'] = df_correlation_train['correlation train']
df['train correlation enformer'] = df_correlation_train_enformer['correlation train enformer']

# calculate mean test and validation correlation score
print(f'mean correlation score test dnn head: {df["test correlation"].mean(axis=0):.4f}')
print(f'mean correlation score validation dnn head: {df["validation correlation"].mean(axis=0):.4f}')
print(f'mean correlation score train dnn head: {df["train correlation"].mean(axis=0):.4f}')
print(f'mean correlation score test enformer: {df["test correlation enformer"].mean(axis=0):.4f}')
print(f'mean correlation score validation enformer: {df["validation correlation enformer"].mean(axis=0):.4f}')
print(f'mean correlation score train enformer: {df["train correlation enformer"].mean(axis=0):.4f}')

print(df)

a = df.groupby('assay type split ChIP')['test correlation'].mean()
print(f'test correlation: {a}\n')

a = df.groupby('assay type split ChIP')['test correlation enformer'].mean()
print(f'test correlation enformer: {a}\n')

a = df.groupby('assay type split ChIP')['train correlation'].mean()
print(f'train correlation: {a}\n')

a = df.groupby('assay type split ChIP')['train correlation enformer'].mean()
print(f'train correlation enformer: {a}\n')

a = df.groupby('assay type split ChIP')['validation correlation'].mean()
print(f'validation correlation: {a}\n')

a = df.groupby('assay type split ChIP')['validation correlation enformer'].mean()
print(f'validation correlation enformer: {a}\n')

sns.plotting_context("paper")
fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, constrained_layout=True, figsize=(10, 3.8))

for i, (key, value) in enumerate(df['assay type split ChIP'].value_counts().to_dict().items()):
    print(i, key)
    ax[i].set(adjustable='box', aspect='equal')
    df_subset = df[df['assay type split ChIP'] == key]
    ax[i].axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
    sns.scatterplot(data = df_subset, x = 'test correlation enformer', y = 'test correlation', color = 'k', ax = ax[i], edgecolor=['white'])
    ax[i].set(xlabel=None)
    ax[i].set(ylabel=None)
    if key == 'DNASE_ATAC': ax[i].set_title(f'DNase, ATAC', fontsize = 9)
    else: ax[i].set_title(f'{key}', fontsize = 9)
    ax[i].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1]) 
    ax[i].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=8)
    ax[i].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1]) 
    ax[i].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=8)
    ax[i].spines[['right', 'top']].set_visible(False)
    #chip TF. hh: 0.504 enf: 0.561
    ax[0].text(0.1, 0.95, '0.504', fontsize = 8, ha='center', va='center', transform=ax[0].transAxes)
    ax[0].text(0.9, 0.03, '0.561', fontsize = 8, ha='center', va='center', transform=ax[0].transAxes)
    # chip histone. hh: 0.642 enf: 0.669
    ax[1].text(0.1, 0.95, '0.643', fontsize = 8, ha='center', va='center', transform=ax[1].transAxes)
    ax[1].text(0.9, 0.03, '0.669', fontsize = 8, ha='center', va='center', transform=ax[1].transAxes)
    # dnase atac. hh: 0.698 enf: 0.832
    ax[2].text(0.1, 0.95, '0.698', fontsize = 8, ha='center', va='center', transform=ax[2].transAxes)
    ax[2].text(0.9, 0.03, '0.832', fontsize = 8, ha='center', va='center', transform=ax[2].transAxes)
    # cage. hh: 0.574 enf: 0.688
    ax[3].text(0.1, 0.95, '0.574', fontsize = 8, ha='center', va='center', transform=ax[3].transAxes)
    ax[3].text(0.9, 0.03, '0.688', fontsize = 8, ha='center', va='center', transform=ax[3].transAxes)
plt.figtext(.5, .17, 'Enformer-pytorch', fontsize=9, ha='center')
fig.supylabel('Human head model', fontsize = 9)
fig.tight_layout()
plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig1_correlation_dnnhead/dnn_head_vs_enformer_perassaytype.png', bbox_inches='tight', dpi = 300)

# train
sns.plotting_context("paper")
fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, constrained_layout=True, figsize=(10, 3.8))
for i, (key, value) in enumerate(df['assay type split ChIP'].value_counts().to_dict().items()):
    print(i, key)
    ax[i].set(adjustable='box', aspect='equal')
    df_subset = df[df['assay type split ChIP'] == key]
    ax[i].axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
    sns.scatterplot(data = df_subset, x = 'train correlation enformer', y = 'train correlation', color = 'k', ax = ax[i], edgecolor=['white'])
    ax[i].set(xlabel=None)
    ax[i].set(ylabel=None)
    if key == 'DNASE_ATAC': ax[i].set_title(f'DNase, ATAC', fontsize = 9)
    else: ax[i].set_title(f'{key}', fontsize = 9)
    ax[i].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1]) 
    ax[i].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=8)
    ax[i].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1]) 
    ax[i].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=8)
    ax[i].spines[['right', 'top']].set_visible(False)
    #chip TF. hh: 0.553 enf: 0.627
    ax[0].text(0.1, 0.95, '0.553', fontsize = 8, ha='center', va='center', transform=ax[0].transAxes)
    ax[0].text(0.9, 0.03, '0.627', fontsize = 8, ha='center', va='center', transform=ax[0].transAxes)
    # chip histone. hh: 0.670 enf: 0.746
    ax[1].text(0.1, 0.95, '0.670', fontsize = 8, ha='center', va='center', transform=ax[1].transAxes)
    ax[1].text(0.9, 0.03, '0.746', fontsize = 8, ha='center', va='center', transform=ax[1].transAxes)
    # dnase atac. hh: 0.721 enf: 0.897
    ax[2].text(0.1, 0.95, '0.721', fontsize = 8, ha='center', va='center', transform=ax[2].transAxes)
    ax[2].text(0.9, 0.03, '0.897', fontsize = 8, ha='center', va='center', transform=ax[2].transAxes)
    # cage. hh: 0.682 enf: 0.928
    ax[3].text(0.1, 0.95, '0.682', fontsize = 8, ha='center', va='center', transform=ax[3].transAxes)
    ax[3].text(0.9, 0.03, '0.928', fontsize = 8, ha='center', va='center', transform=ax[3].transAxes)
plt.figtext(.5, .17, 'Enformer-pytorch', fontsize=9, ha='center')
fig.supylabel('Human head model', fontsize = 9)
fig.tight_layout()
plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig1_correlation_dnnhead/dnn_head_vs_enformer_perassaytype_train.png', bbox_inches='tight', dpi = 300)

# valid
sns.plotting_context("paper")
fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, constrained_layout=True, figsize=(10, 3.8))

for i, (key, value) in enumerate(df['assay type split ChIP'].value_counts().to_dict().items()):
    print(i, key)
    ax[i].set(adjustable='box', aspect='equal')
    df_subset = df[df['assay type split ChIP'] == key]
    ax[i].axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
    sns.scatterplot(data = df_subset, x = 'validation correlation enformer', y = 'validation correlation', color = 'k', ax = ax[i], edgecolor=['white'])
    ax[i].set(xlabel=None)
    ax[i].set(ylabel=None)
    if key == 'DNASE_ATAC': ax[i].set_title(f'DNase, ATAC', fontsize = 9)
    else: ax[i].set_title(f'{key}', fontsize = 9)
    ax[i].set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1]) 
    ax[i].set_xticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=8)
    ax[i].set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1]) 
    ax[i].set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], fontsize=8)
    ax[i].spines[['right', 'top']].set_visible(False)

    #chip TF. hh: 0.481 enf: 0.536
    ax[0].text(0.1, 0.95, '0.481', fontsize = 8, ha='center', va='center', transform=ax[0].transAxes)
    ax[0].text(0.9, 0.03, '0.536', fontsize = 8, ha='center', va='center', transform=ax[0].transAxes)

    # chip histone. hh: 0.618 enf: 0.641
    ax[1].text(0.1, 0.95, '0.618', fontsize = 8, ha='center', va='center', transform=ax[1].transAxes)
    ax[1].text(0.9, 0.03, '0.641', fontsize = 8, ha='center', va='center', transform=ax[1].transAxes)

    # dnase atac. hh: 0.608 enf: 0.775
    ax[2].text(0.1, 0.95, '0.608', fontsize = 8, ha='center', va='center', transform=ax[2].transAxes)
    ax[2].text(0.9, 0.03, '0.775', fontsize = 8, ha='center', va='center', transform=ax[2].transAxes)

    # cage. hh: 0.588 enf:  0.705
    ax[3].text(0.1, 0.95, '0.588', fontsize = 8, ha='center', va='center', transform=ax[3].transAxes)
    ax[3].text(0.9, 0.03, ' 0.705', fontsize = 8, ha='center', va='center', transform=ax[3].transAxes)

plt.figtext(.5, .17, 'Enformer-pytorch', fontsize=9, ha='center')
# fig.supxlabel('Enformer-pytorch')
fig.supylabel('Human head model', fontsize = 9)

fig.tight_layout()
plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig1_correlation_dnnhead/dnn_head_vs_enformer_perassaytype_valid.png', bbox_inches='tight', dpi = 300)