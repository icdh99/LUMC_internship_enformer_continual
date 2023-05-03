import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""
Compare performance of 5313  tracks to dnn head model
"""

# input_file = '/exports/humgen/idenhond/data/Basenji/human-targets.txt'
# df = pd.read_csv(input_file, sep = '\t')
# df[['assay type', 'description2']] = df.description.str.split(':', n = 1, expand = True) # make new column for assay type
# def f(row):
#     if row['assay type'] == 'CHIP':
#         if any(row['description2'].startswith(x) for x in ['H2AK', 'H2BK', 'H3K', 'H4K']): val = 'ChIP Histone'
#         else: val = 'ChIP TF'
#     elif row['assay type'] == 'DNASE' or row['assay type'] == 'ATAC': val = 'DNASE/ATAC'
#     else: val = row['assay type']
#     return val
# df['assay type split ChIP'] = df.apply(f, axis=1)
# print(f'Number of tracks: {df.shape[0]}\n')
# print(f"Number of trakcs per assay type: \n {df['assay type'].value_counts()}")
# print(f"Number of trakcs per assay type: \n {df['assay type split ChIP'].value_counts()}\n")

# # read csv with correlation score per track for test and validation sequences
# col_name = ['correlation test']
# df_correlation_test = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_alltracks_2404.csv', index_col = 0, names = col_name)
# print(f'df correlation test all tracks shape: {df_correlation_test.shape}')
# col_name = ['correlation test enformer']
# df_correlation_test_enformer = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_own_output_pretrainedmodel.csv', index_col = 0, names = col_name)
# col_name = ['correlation test dnn head']
# df_correlation_test_dnnhead = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_dnn_head.csv', index_col = 0, names = col_name)
# print(f'df correlation dnn head shape: {df_correlation_test_dnnhead.shape}')
# col_name = ['correlation validation']
# df_correlation_validation = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_alltracks_2404.csv', index_col = 0, names = col_name)
# col_name = ['correlation valid enformer']
# df_correlation_validation_enformer = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_own_output_pretrainedmodel.csv', index_col = 0, names = col_name)
# # col_name = ['correlation train']
# # df_correlation_train = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train_dnn_head.csv', index_col = 0, names = col_name)
# # col_name = ['correlation train enformer']
# # df_correlation_train_enformer = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train.csv', index_col = 0, names = col_name)

# df_correlation_test = df_correlation_test.tail(-1)
# df_correlation_test_enformer = df_correlation_test_enformer.tail(-1)
# df_correlation_test_dnnhead = df_correlation_test_dnnhead.tail(-1)
# df_correlation_validation = df_correlation_validation.tail(-1)
# df_correlation_validation_enformer = df_correlation_validation_enformer.tail(-1)
# # df_correlation_train = df_correlation_train.tail(-1)
# # df_correlation_train_enformer = df_correlation_train_enformer.tail(-1)

# # add column to df with test and validation correlation scores
# df['test correlation'] = df_correlation_test['correlation test']
# df['test correlation dnn head'] = df_correlation_test_dnnhead['correlation test dnn head']
# df['test correlation enformer'] = df_correlation_test_enformer['correlation test enformer']
# df['validation correlation'] = df_correlation_validation['correlation validation']
# df['validation correlation enformer'] = df_correlation_validation_enformer['correlation valid enformer']
# # df['train correlation'] = df_correlation_train['correlation train']
# # df['train correlation enformer'] = df_correlation_train_enformer['correlation train enformer']

# # calculate mean test and validation correlation score
# print(f'mean correlation score test all tracks model: {df["test correlation"].mean(axis=0):.4f}')
# print(f'mean correlation score test enformer: {df["test correlation enformer"].mean(axis=0):.4f}')
# print(f'mean correlation score test dnn head: {df["test correlation dnn head"].mean(axis=0):.4f}')
# print(f'mean correlation score validation: {df["validation correlation"].mean(axis=0):.4f}')
# print(f'mean correlation score test enformer: {df["test correlation enformer"].mean(axis=0):.4f}')


# print(df)
# plt.figure(1)
# ax = sns.boxplot(data = df[['test correlation', 'test correlation dnn head', 'test correlation enformer']], showmeans = True)
# ax.set_xticklabels([f'All tracks model\n{df["test correlation"].mean(axis=0):.4f}', 
#                     f'DNN head\n{df["test correlation dnn head"].mean(axis=0):.4f}', 
#                     f'Enformer\n{df["test correlation enformer"].mean(axis=0):.4f}'])
# plt.ylabel('Pearson Correlation Coefficient')
# plt.title(f'Test set correlation for 5313 tracks')
# plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/alltracks_2404/alltracks_2404_boxplot_test_vsenformeranddnnhead_corr.png', bbox_inches='tight')
# plt.close()

# for key, value in df['assay type split ChIP'].value_counts().to_dict().items():
#     df_subset = df[df['assay type split ChIP'] == key]
#     print(key, value)
#     plt.figure()
#     plt.title(f'Assay type: {key}')
#     plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
#     sns.scatterplot(data = df_subset, x = 'test correlation', y = 'test correlation dnn head', color = 'k')
#     plt.ylabel('Test correlation DNN head')
#     plt.xlabel('Test correlation all tracks model')
#     if key == 'DNASE/ATAC':
#         key = 'DNASE_ATAC'
#     plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/Plots/alltracks_2404/alltracks_2404_scatterplot_test_corr_dnnhead_{key}.png', bbox_inches='tight')
#     plt.close()


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

print(df_correlation_validation_alltracks.shape)
print(df_correlation_validation_newtracks.shape)

print(df_correlation_test_newtracks)
print(df_correlation_test_alltracks)

df['correlation test all tracks model'] = df_correlation_test_alltracks['correlation test all tracks model']
df['correlation test new tracks model'] = df_correlation_test_newtracks['correlation test new tracks model']
df['correlation validation all tracks model'] = df_correlation_validation_alltracks['correlation validation all tracks model']
df['correlation validation new tracks model'] = df_correlation_validation_newtracks['correlation validation new tracks model']

print(df)

plt.figure()
ax = sns.boxplot(data = df[['correlation test all tracks model', 'correlation test new tracks model']], showmeans = True)
ax.set_xticklabels([f'All tracks model\n{df["correlation test all tracks model"].mean(axis=0):.4f}', 
                    f'New tracks model\n{df["correlation test new tracks model"].mean(axis=0):.4f}'])
plt.ylabel('Pearson Correlation Coefficient')
plt.title(f'Test set correlation for 27 new tracks')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/alltracks_2404/alltracks_2404_boxplot_test_newtracks.png', bbox_inches='tight')
plt.close()

plt.figure()
ax = sns.boxplot(data = df[['correlation test all tracks model', 'correlation test new tracks model']], showmeans = True, orient='h')
ax.set_yticklabels([f'All tracks model\n{df["correlation test all tracks model"].mean(axis=0):.4f}', 
                    f'New tracks model\n{df["correlation test new tracks model"].mean(axis=0):.4f}'])
plt.yticks(rotation = 90)
plt.xlabel('Pearson Correlation Coefficient')
plt.title(f'Test set correlation for 27 new tracks')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/alltracks_2404/alltracks_2404_boxplot_test_newtracks_horizontal.png', bbox_inches='tight')
plt.close()

plt.figure()
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df, x = 'correlation test all tracks model', y = 'correlation test all tracks model', hue = 'assay type')
plt.xlabel('New tracks model')
plt.ylabel('All tracks model')
plt.title(f'Test set correlation for 27 new tracks')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/alltracks_2404/alltracks_2404_scatter_test_newtracks.png', bbox_inches='tight')
plt.close()

df_melt = pd.melt(df, id_vars = ['identifier', 'assay type'], value_vars = ['correlation test new tracks model', 'correlation test all tracks model'], var_name = 'correlation model')
print(df_melt)

plt.figure()
ax = sns.boxplot(data = df_melt, x = 'correlation model', y = 'value', hue = 'assay type', showmeans = False, width = 0.8)
ax.set_xticklabels([f'All tracks model\n{df["correlation test all tracks model"].mean(axis=0):.4f}', 
                    f'New tracks model\n{df["correlation test new tracks model"].mean(axis=0):.4f}'])
plt.ylabel('Pearson Correlation Coefficient')
plt.title(f'Test set correlation for 27 new tracks')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/alltracks_2404/alltracks_2404_boxplot_test_newtracks_assaytype.png', bbox_inches='tight')
plt.close()

with sns.plotting_context("poster"):
    plt.figure()
    plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
    sns.scatterplot(data = df, x = 'correlation test all tracks model', y = 'correlation test all tracks model', hue = 'assay type')
    plt.xlabel('New tracks model')
    plt.ylabel('All tracks model')
    plt.title(f'Test set correlation for 27 new tracks')
    plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/alltracks_2404/alltracks_2404_scatter_test_newtracks_poster.png', bbox_inches='tight')
    plt.close()

with sns.plotting_context("poster"):
    plt.figure(figsize = (6.4*1.5, 6.8*1.5))
    fig_width, fig_height = plt.gcf().get_size_inches()
    print(fig_width, fig_height) # 6.4 4.8
    plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
    sns.scatterplot(data = df, x = 'correlation test all tracks model', y = 'correlation test all tracks model', hue = 'assay type',  s = 200, alpha = 0.8)
    plt.xlabel('New tracks model')
    plt.ylabel('All tracks model')
    plt.legend(title = 'Assay type')
    plt.title(f'Test set correlation for 27 new tracks')
    plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/alltracks_2404/alltracks_2404_scatter_test_newtracks_poster.png', bbox_inches='tight')
    plt.savefig('/exports/humgen/idenhond/plots_poster_biosb/alltracks_2404_scatter_test_newtracks_poster.png', bbox_inches='tight')
    plt.close()


with sns.plotting_context("poster"):
    plt.figure(figsize = (6.4*1.5, 6.8*1.5))
    sns.pointplot(data = df_melt, x = 'correlation model', y = 'value', hue = 'assay type')
    plt.savefig('/exports/humgen/idenhond/plots_poster_biosb/alltracks_2404_pointplot_test_newtracks_poster.png', bbox_inches='tight')
