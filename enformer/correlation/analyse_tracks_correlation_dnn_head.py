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


# select 10 tracks with highest test correlation score
    # TODO ..... 

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

# plt.figure(1)
# sns.boxplot(data = df[['test correlation', 'test correlation enformer']], showmeans = True)
# plt.ylabel('Pearson Correlation Coefficient')
# plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head/dnn_head_boxplot_test_vsenformer_corr.png', bbox_inches='tight')

# plt.figure(2)
# sns.boxplot(data = df[['validation correlation', 'validation correlation enformer']], showmeans = True)
# plt.ylabel('Pearson Correlation Coefficient')
# plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head/dnn_head_boxplot_valid_vsenformer_corr.png', bbox_inches='tight')

# plt.figure(3)
# sns.boxplot(data = df[['train correlation', 'train correlation enformer']], showmeans = True)
# plt.ylabel('Pearson Correlation Coefficient')
# plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head/dnn_head_boxplot_train_vsenformer_corr.png', bbox_inches='tight')

# plt.figure(4)
# sns.boxplot(data = df[['test correlation', 'validation correlation']], showmeans = True)
# plt.ylabel('Pearson Correlation Coefficient')
# plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head/dnn_head_boxplot_test_valid_corr.png', bbox_inches='tight')

# plt.figure(5)
# sns.boxplot(data = df[['train correlation', 'test correlation']], showmeans = True)
# plt.ylabel('Pearson Correlation Coefficient')
# plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head/dnn_head_boxplot_test_train_corr.png', bbox_inches='tight')

# plt.figure(6)
# plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
# sns.scatterplot(data = df, x = 'test correlation enformer', y = 'test correlation')
# plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head/dnn_head_scatterplot_test_enformer_corr.png', bbox_inches='tight')

# plt.figure(7)
# plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
# sns.scatterplot(data = df, x = 'test correlation enformer', y = 'test correlation', hue = 'assay type')
# plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head/dnn_head_scatterplot_test_enformer_corr_asssaytype.png', bbox_inches='tight')

plt.figure(7, figsize = (12, 10))
with sns.plotting_context("talk"):
    plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
    sns.scatterplot(data = df, x = 'test correlation enformer', y = 'test correlation', hue = 'assay type split ChIP')
    # plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title = 'Assay type')
    plt.legend(bbox_to_anchor=(0, 0.99), loc="upper left", title = 'Assay type')
    # plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", title = 'Assay type', ncols = 4, mode = 'expand')
    plt.xlabel('Pearson correlation (Enformer model)')
    plt.ylabel('Pearson correlation (Our model)')
    plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head/dnn_head_scatterplot_test_enformer_corr_asssaytypechip_talk.png', bbox_inches='tight')

plt.figure(8, figsize = (12, 10))
with sns.plotting_context("poster"):
    plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
    sns.scatterplot(data = df, x = 'test correlation enformer', y = 'test correlation', hue = 'assay type split ChIP')
    # plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", title = 'Assay type')
    plt.legend(bbox_to_anchor=(0, 0.99), loc="upper left", title = 'Assay type')
    # plt.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="lower left", title = 'Assay type', ncols = 4, mode = 'expand')
    plt.xlabel('Pearson correlation (Enformer model)')
    plt.ylabel('Pearson correlation (Our model)')
    plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head/dnn_head_scatterplot_test_enformer_corr_asssaytypechip_poster.png', bbox_inches='tight')

with sns.plotting_context("poster"):
    for key, value in df['assay type split ChIP'].value_counts().to_dict().items():
        df_subset = df[df['assay type split ChIP'] == key]
        plt.figure()
        if key == 'DNASE_ATAC':
            plt.title('DNASE, ATAC')
        else: plt.title(f'{key}')
        plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
        if key == 'DNASE_ATAC':
            plt.tick_params(axis='x',          # changes apply to the x-axis, 
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False) # labels along the bottom edge are off
            plt.ylabel('Correlation (Our model)')
        if key == 'CAGE':
            plt.tick_params(axis='y',          # changes apply to the x-axis, 
                            which='both',      # both major and minor ticks are affected
                            left=False,      # ticks along the bottom edge are off
                            right=False,         # ticks along the top edge are off
                            labelleft=False) # labels along the bottom edge are off
            plt.tick_params(axis='x',          # changes apply to the x-axis, 
                            which='both',      # both major and minor ticks are affected
                            bottom=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelbottom=False) # labels along the bottom edge are off
            # plt.ylabel('Correlation (Our model)')
            plt.xlabel(None)
            plt.ylabel(None)
        if key == 'ChIP TF':
            # plt.tick_params(axis='x',          # changes apply to the x-axis, 
            #                 which='both',      # both major and minor ticks are affected
            #                 bottom=False,      # ticks along the bottom edge are off
            #                 top=False,         # ticks along the top edge are off
            #                 labelbottom=False) # labels along the bottom edge are off
            plt.ylabel('Correlation (Our model)')
            plt.xlabel('Correlation (Enformer model)')
        if key == 'ChIP Histone':
            plt.tick_params(axis='y',          # changes apply to the x-axis, 
                            which='both',      # both major and minor ticks are affected
                            left=False,      # ticks along the bottom edge are off
                            top=False,         # ticks along the top edge are off
                            labelleft=False) # labels along the bottom edge are off
            plt.xlabel('Correlation (Enformer model)')
            plt.ylabel(None)
        sns.scatterplot(data = df_subset, x = 'test correlation enformer', y = 'test correlation', color = 'k')
        plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head/dnn_head_pretrainedmodel_scatterplot_test_enformer_corr_poster_{key}.png', bbox_inches='tight')

with sns.plotting_context("talk"):
    for key, value in df['assay type split ChIP'].value_counts().to_dict().items():
        df_subset = df[df['assay type split ChIP'] == key]
        plt.figure()
        if key == 'DNASE_ATAC':
            plt.title('DNASE, ATAC')
        else: plt.title(f'{key}')
        plt.xlabel('Correlation (Enformer model)')
        plt.ylabel('Correlation (Our model)')
        plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
        sns.scatterplot(data = df_subset, x = 'test correlation enformer', y = 'test correlation', color = 'k')
        plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head/dnn_head_pretrainedmodel_scatterplot_test_enformer_corr_talk_{key}.png', bbox_inches='tight')

exit()
plt.figure(8)
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df, x = 'validation correlation enformer', y = 'validation correlation')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head/dnn_head_scatterplot_valid_enformer_corr.png', bbox_inches='tight')

plt.figure(9)
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df, x = 'validation correlation enformer', y = 'validation correlation', hue = 'assay type')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head/dnn_head_scatterplot_valid_enformer_corr_asssaytype.png', bbox_inches='tight')

plt.figure(10)
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df, x = 'train correlation enformer', y = 'train correlation')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head/dnn_head_scatterplot_train_enformer_corr.png', bbox_inches='tight')

plt.figure(11)
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df, x = 'train correlation enformer', y = 'train correlation', hue = 'assay type')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head/dnn_head_scatterplot_train_enformer_corr_asssaytype.png', bbox_inches='tight')

plt.figure(12)
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df, x = 'test correlation', y = 'validation correlation', hue = 'assay type')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head/dnn_head_scatterplot_valid_test_corr_asssaytype.png', bbox_inches='tight')

for key, value in df['assay type'].value_counts().to_dict().items():
    df_subset = df[df['assay type'] == key]
    plt.figure()
    plt.title(f'Assay type: {key}')
    plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
    sns.scatterplot(data = df_subset, x = 'test correlation', y = 'validation correlation')
    plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head/dnn_head_pretrainedmodel_scatterplot_test_val_corr_{key}.png', bbox_inches='tight')

# enformer test vs test for each assay type

for key, value in df['assay type'].value_counts().to_dict().items():
    df_subset = df[df['assay type'] == key]
    plt.figure()
    plt.title(f'Assay type: {key}')
    plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
    sns.scatterplot(data = df_subset, x = 'test correlation enformer', y = 'test correlation')
    plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head/dnn_head_pretrainedmodel_scatterplot_test_enformer_corr_{key}.png', bbox_inches='tight')

for key, value in df['assay type'].value_counts().to_dict().items():
    df_subset = df[df['assay type'] == key]
    plt.figure()
    plt.title(f'Assay type: {key}')
    plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
    sns.scatterplot(data = df_subset, x = 'validation correlation enformer', y = 'validation correlation')
    plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head/dnn_head_pretrainedmodel_scatterplot_valid_enformer_corr_{key}.png', bbox_inches='tight')

for key, value in df['assay type'].value_counts().to_dict().items():
    df_subset = df[df['assay type'] == key]
    plt.figure()
    plt.title(f'Assay type: {key}')
    plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
    sns.scatterplot(data = df_subset, x = 'train correlation enformer', y = 'train correlation')
    plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_head/dnn_head_pretrainedmodel_scatterplot_train_enformer_corr_{key}.png', bbox_inches='tight')
