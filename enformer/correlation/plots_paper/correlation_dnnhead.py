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

# sns.plotting_context("paper")

fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, constrained_layout=True)
fig.tight_layout()
plt.xlabel('Enformer-pytorch')
plt.ylabel('Human head model')

for i, (key, value) in enumerate(df['assay type split ChIP'].value_counts().to_dict().items()):
    print(i, key)
    ax[i].set(adjustable='box', aspect='equal')
    df_subset = df[df['assay type split ChIP'] == key]
    
    ax[i].axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')

    sns.scatterplot(data = df_subset, x = 'test correlation enformer', y = 'test correlation', color = 'k', ax = ax[i])

plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots/Fig1_correlation_dnnhead/dnn_head_vs_enformer_perassaytype.png', bbox_inches='tight')
