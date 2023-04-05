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
    elif row['assay type'] == 'DNASE' or row['assay type'] == 'ATAC': val = 'DNASE/ATAC'
    else: val = row['assay type']
    return val
df['assay type split ChIP'] = df.apply(f, axis=1)
print(f'Number of tracks: {df.shape[0]}\n')
print(f"Number of trakcs per assay type: \n {df['assay type'].value_counts()}")
print(f"Number of trakcs per assay type: \n {df['assay type split ChIP'].value_counts()}\n")

# select 10 tracks with highest test correlation score
    # TODO ..... 

# read csv with correlation score per track for test and validation sequences
col_name = ['correlation test']
df_correlation_test = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_dnn_head_alltracks.csv', index_col = 0, names = col_name)
print(f'df correlation test all tracks shape: {df_correlation_test.shape}')
col_name = ['correlation test enformer']
df_correlation_test_enformer = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_own_output_pretrainedmodel.csv', index_col = 0, names = col_name)
col_name = ['correlation test dnn head']
df_correlation_test_dnnhead = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_dnn_head.csv', index_col = 0, names = col_name)
print(f'df correlation dnn head shape: {df_correlation_test_dnnhead.shape}')
col_name = ['correlation validation']
df_correlation_validation = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_dnn_head_alltracks.csv', index_col = 0, names = col_name)
col_name = ['correlation valid enformer']
df_correlation_validation_enformer = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_own_output_pretrainedmodel.csv', index_col = 0, names = col_name)
# col_name = ['correlation train']
# df_correlation_train = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train_dnn_head.csv', index_col = 0, names = col_name)
# col_name = ['correlation train enformer']
# df_correlation_train_enformer = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train.csv', index_col = 0, names = col_name)

df_correlation_test = df_correlation_test.tail(-1)
df_correlation_test_enformer = df_correlation_test_enformer.tail(-1)
df_correlation_test_dnnhead = df_correlation_test_dnnhead.tail(-1)
df_correlation_validation = df_correlation_validation.tail(-1)
df_correlation_validation_enformer = df_correlation_validation_enformer.tail(-1)
# df_correlation_train = df_correlation_train.tail(-1)
# df_correlation_train_enformer = df_correlation_train_enformer.tail(-1)

# add column to df with test and validation correlation scores
df['test correlation'] = df_correlation_test['correlation test']
df['test correlation dnn head'] = df_correlation_test_dnnhead['correlation test dnn head']
df['test correlation enformer'] = df_correlation_test_enformer['correlation test enformer']
df['validation correlation'] = df_correlation_validation['correlation validation']
df['validation correlation enformer'] = df_correlation_validation_enformer['correlation valid enformer']
# df['train correlation'] = df_correlation_train['correlation train']
# df['train correlation enformer'] = df_correlation_train_enformer['correlation train enformer']

# calculate mean test and validation correlation score
print(f'mean correlation score test all tracks model: {df["test correlation"].mean(axis=0):.4f}')
print(f'mean correlation score test enformer: {df["test correlation enformer"].mean(axis=0):.4f}')
print(f'mean correlation score test dnn head: {df["test correlation dnn head"].mean(axis=0):.4f}')
print(f'mean correlation score validation: {df["validation correlation"].mean(axis=0):.4f}')
print(f'mean correlation score test enformer: {df["test correlation enformer"].mean(axis=0):.4f}')

# print(f'mean correlation score train dnn head: {df["train correlation"].mean(axis=0):.4f}')


# plt.figure(1)
# sns.boxplot(data = df[['test correlation', 'test correlation enformer']], showmeans = True)
# plt.ylabel('Pearson Correlation Coefficient')
# plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head_alltracks/dnn_head_alltracks_boxplot_test_vsenformer_corr.png', bbox_inches='tight')

print(df)
plt.figure(1)
ax = sns.boxplot(data = df[['test correlation', 'test correlation dnn head', 'test correlation enformer']], showmeans = True)
ax.set_xticklabels(['All tracks model', 'DNN head', 'Enformer'])
plt.ylabel('Pearson Correlation Coefficient')
plt.title(f'Test set correlation for 5313 tracks')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head_alltracks/dnn_head_alltracks_boxplot_test_vsenformeranddnnhead_corr.png', bbox_inches='tight')


for key, value in df['assay type split ChIP'].value_counts().to_dict().items():
    df_subset = df[df['assay type split ChIP'] == key]
    print(key, value)
    plt.figure()
    plt.title(f'Assay type: {key}')
    plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
    sns.scatterplot(data = df_subset, x = 'test correlation', y = 'test correlation dnn head', color = 'k')
    plt.ylabel('Test correlation DNN head')
    plt.xlabel('Test correlation all tracks model')
    if key == 'DNASE/ATAC':
        key = 'DNASE_ATAC'
    plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head_alltracks/dnn_head_alltracks_scatterplot_test_corr_dnnhead_{key}.png', bbox_inches='tight')

exit()
plt.figure(2)
sns.boxplot(data = df[['validation correlation', 'validation correlation enformer']], showmeans = True)
plt.ylabel('Pearson Correlation Coefficient')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head_alltracks/dnn_head_alltracks_boxplot_valid_vsenformer_corr.png', bbox_inches='tight')

# plt.figure(3)
# sns.boxplot(data = df[['train correlation', 'train correlation enformer']], showmeans = True)
# plt.ylabel('Pearson Correlation Coefficient')
# plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head_alltracks/dnn_head_alltracks_boxplot_train_vsenformer_corr.png', bbox_inches='tight')

plt.figure(4)
sns.boxplot(data = df[['test correlation', 'validation correlation']], showmeans = True)
plt.ylabel('Pearson Correlation Coefficient')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head_alltracks/dnn_head_alltracks_boxplot_test_valid_corr.png', bbox_inches='tight')

# plt.figure(5)
# sns.boxplot(data = df[['train correlation', 'test correlation']], showmeans = True)
# plt.ylabel('Pearson Correlation Coefficient')
# plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head_alltracks/dnn_head_alltracks_boxplot_test_train_corr.png', bbox_inches='tight')

plt.figure(6)
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df, x = 'test correlation enformer', y = 'test correlation')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head_alltracks/dnn_head_alltracks_scatterplot_test_enformer_corr.png', bbox_inches='tight')

plt.figure(7)
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df, x = 'test correlation enformer', y = 'test correlation', hue = 'assay type')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head_alltracks/dnn_head_alltracks_scatterplot_test_enformer_corr_asssaytype.png', bbox_inches='tight')

plt.figure(8)
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df, x = 'validation correlation enformer', y = 'validation correlation')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head_alltracks/dnn_head_alltracks_scatterplot_valid_enformer_corr.png', bbox_inches='tight')

plt.figure(9)
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df, x = 'validation correlation enformer', y = 'validation correlation', hue = 'assay type')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head_alltracks/dnn_head_alltracks_scatterplot_valid_enformer_corr_asssaytype.png', bbox_inches='tight')

# plt.figure(10)
# plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
# sns.scatterplot(data = df, x = 'train correlation enformer', y = 'train correlation')
# plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head_alltracks/dnn_head_alltracks_scatterplot_train_enformer_corr.png', bbox_inches='tight')

# plt.figure(11)
# plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
# sns.scatterplot(data = df, x = 'train correlation enformer', y = 'train correlation', hue = 'assay type')
# plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head_alltracks/dnn_head_alltracks_scatterplot_train_enformer_corr_asssaytype.png', bbox_inches='tight')

plt.figure(12)
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df, x = 'test correlation', y = 'validation correlation', hue = 'assay type')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head_alltracks/dnn_head_alltracks_scatterplot_valid_test_corr_asssaytype.png', bbox_inches='tight')

for key, value in df['assay type'].value_counts().to_dict().items():
    df_subset = df[df['assay type'] == key]
    plt.figure()
    plt.title(f'Assay type: {key}')
    plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
    sns.scatterplot(data = df_subset, x = 'test correlation', y = 'validation correlation')
    plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head_alltracks/dnn_head_alltracks_pretrainedmodel_scatterplot_test_val_corr_{key}.png', bbox_inches='tight')

# enformer test vs test for each assay type

for key, value in df['assay type'].value_counts().to_dict().items():
    df_subset = df[df['assay type'] == key]
    plt.figure()
    plt.title(f'Assay type: {key}')
    plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
    sns.scatterplot(data = df_subset, x = 'test correlation enformer', y = 'test correlation')
    plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head_alltracks/dnn_head_alltracks_pretrainedmodel_scatterplot_test_enformer_corr_{key}.png', bbox_inches='tight')

for key, value in df['assay type'].value_counts().to_dict().items():
    df_subset = df[df['assay type'] == key]
    plt.figure()
    plt.title(f'Assay type: {key}')
    plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
    sns.scatterplot(data = df_subset, x = 'validation correlation enformer', y = 'validation correlation')
    plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head_alltracks/dnn_head_alltracks_pretrainedmodel_scatterplot_valid_enformer_corr_{key}.png', bbox_inches='tight')

# for key, value in df['assay type'].value_counts().to_dict().items():
#     df_subset = df[df['assay type'] == key]
#     plt.figure()
#     plt.title(f'Assay type: {key}')
#     plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
#     sns.scatterplot(data = df_subset, x = 'train correlation enformer', y = 'train correlation')
#     plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head_alltracks/dnn_head_alltracks_pretrainedmodel_scatterplot_train_enformer_corr_{key}.png', bbox_inches='tight')
