import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# file with targets & split assay type 
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
print(f'Number of tracks: {df.shape[0]}')
print(f"Number of trakcs per assay type: \n {df['assay type split ChIP'].value_counts()}\n")

df_correlation_test_old = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_dnn_head.csv', index_col = 0, names = ['correlation test old']).tail(-1)
df_correlation_test_new = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_dnn_head_retrain2703.csv', index_col = 0, names = ['correlation test new']).tail(-1)
df_correlation_test_enformer = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_own_output_pretrainedmodel.csv', index_col = 0, names = ['correlation test enformer']).tail(-1)
df_correlation_valid_old = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_dnn_head.csv', index_col = 0, names = ['correlation valid old']).tail(-1)
df_correlation_valid_new = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_dnn_head_retrain2703.csv', index_col = 0, names = ['correlation valid new']).tail(-1)
df_correlation_valid_enformer = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_own_output_pretrainedmodel.csv', index_col = 0, names = ['correlation valid enformer']).tail(-1)
df_correlation_train_old = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train_dnn_head.csv', index_col = 0, names = ['correlation train old']).tail(-1)
df_correlation_train_new = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train_dnn_head_retrain2703.csv', index_col = 0, names = ['correlation train new']).tail(-1)
df_correlation_train_enformer = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train.csv', index_col = 0, names = ['correlation train enformer']).tail(-1)

df['test correlation old'] = df_correlation_test_old['correlation test old']
df['test correlation new'] = df_correlation_test_new['correlation test new']
df['test correlation enformer'] = df_correlation_test_enformer['correlation test enformer']

df['valid correlation old'] = df_correlation_valid_old['correlation valid old']
df['valid correlation new'] = df_correlation_valid_new['correlation valid new']
df['valid correlation enformer'] = df_correlation_valid_enformer['correlation valid enformer']

df['train correlation old'] = df_correlation_train_old['correlation train old']
df['train correlation new'] = df_correlation_train_new['correlation train new']
df['train correlation enformer'] = df_correlation_train_enformer['correlation train enformer']

print(df)


for subset in ['test', 'valid', 'train']:
    plt.figure()
    plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
    sns.scatterplot(data = df, x = f'{subset} correlation old', y = f'{subset} correlation new', hue = 'assay type split ChIP')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/Plots/compare_dnnhead_retrain/scatterplot_{subset}_dnnhead_oldnew.png', bbox_inches='tight')
    plt.close()


for subset in ['test', 'valid', 'train']:
    for key, value in df['assay type split ChIP'].value_counts().to_dict().items():
        df_subset = df[df['assay type split ChIP'] == key]
        plt.figure()
        plt.title(f'Assay type: {key}')
        # mean = df_subset[f"{subset} correlation"].mean(axis=0)
        # mean_enformer = df_subset[f"{subset} correlation enformer"].mean(axis=0)
        # print(mean)
        # print(mean_enformer)
        # plt.axhline(y = mean, linewidth = 1, color = 'k', label = f'mean dnn head: {mean:.3f}')
        # plt.axvline(x = mean_enformer, linewidth = 1, color = 'g', label = f'mean enformer: {mean_enformer:.3f}')
        plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
        sns.scatterplot(data = df_subset, x = f'{subset} correlation enformer', y = f'{subset} correlation new')
        plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        if key == 'DNASE/ATAC': key = 'DNASE_ATAC'
        plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/Plots/compare_dnnhead_retrain/scatterplot_{subset}_enformer_corr_{key}.png', bbox_inches='tight')
        plt.close()