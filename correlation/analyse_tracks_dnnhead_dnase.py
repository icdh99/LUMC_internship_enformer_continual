import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_file =  '/exports/humgen/idenhond/data/Basenji/human-targets.txt'
df = pd.read_csv(input_file, sep = '\t')
df[['assay type', 'description2']] = df.description.str.split(':', n = 1, expand = True) # make new column for assay type
print(f'Number of tracks: {df.shape[0]}')
print(f"Number of trakcs per assay type: \n {df['assay type'].value_counts()}\n")
df = df.head(674)

df_correlation_test = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_dnase_3103.csv', index_col = 0, names = ['correlation test']).tail(-1)
df_correlation_valid = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_dnase_3103.csv', index_col = 0, names = ['correlation valid']).tail(-1)
df_correlation_train = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train_dnase_3103.csv', index_col = 0, names = ['correlation train']).tail(-1)

df_correlation_test_dnnhead = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_dnn_head.csv', index_col = 0, names = ['correlation test dnn head']).tail(-1).head(674)
df_correlation_valid_dnnhead = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_dnn_head.csv', index_col = 0, names = ['correlation valid dnn head']).tail(-1).head(674)
df_correlation_train_dnnhead = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train_dnn_head.csv', index_col = 0, names = ['correlation train dnn head']).tail(-1).head(674)

df_correlation_test_enformer = pd.read_csv(f'/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test.csv', index_col = 0, names = ['correlation test enformer']).tail(-1).head(674)
print(df_correlation_test_enformer.shape)
print(df_correlation_test_enformer)

df['test correlation'] = df_correlation_test['correlation test']
df['test correlation dnn head'] = df_correlation_test_dnnhead['correlation test dnn head']
df['valid correlation'] = df_correlation_valid['correlation valid']
df['valid correlation dnn head'] = df_correlation_valid_dnnhead['correlation valid dnn head']
df['train correlation'] = df_correlation_train['correlation train']
df['train correlation dnn head'] = df_correlation_train_dnnhead['correlation train dnn head']
df['test correlation enformer'] = df_correlation_test_enformer['correlation test enformer']

print(f'mean correlation score test: {df["test correlation"].mean(axis=0):.4f}')
print(f'mean correlation score test dnn head: {df["test correlation dnn head"].mean(axis=0):.4f}')
print(f'mean correlation score valid: {df["valid correlation"].mean(axis=0):.4f}')
print(f'mean correlation score valid dnn head: {df["valid correlation dnn head"].mean(axis=0):.4f}')
print(f'mean correlation score train: {df["train correlation"].mean(axis=0):.4f}')
print(f'mean correlation score train dnn head: {df["train correlation dnn head"].mean(axis=0):.4f}')
print(f'mean correlation score test enformer: {df["test correlation enformer"].mean(axis=0):.4f}')

print(df)

plt.figure()
ax = sns.boxplot(data = df[[ 'test correlation dnn head', 'test correlation']], showmeans = True)
ax.set_xticklabels([f'DNN head model\n{df["test correlation dnn head"].mean(axis=0):.4f}', f'DNASE model\n{df["test correlation"].mean(axis=0):.4f}'])
plt.ylabel('Pearson Correlation Coefficient')
plt.title(f'Test set correlation for DNASE tracks')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnase_model/dnase_model_boxplot_test_corr.png', bbox_inches='tight')
plt.close()

plt.figure()
ax = sns.boxplot(data = df[[ 'valid correlation dnn head', 'valid correlation']], showmeans = True)
ax.set_xticklabels([f'DNN head model\n{df["valid correlation dnn head"].mean(axis=0):.4f}', f'DNASE model\n{df["valid correlation"].mean(axis=0):.4f}'])
plt.ylabel('Pearson Correlation Coefficient')
plt.title(f'Valid set correlation for DNASE tracks')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnase_model/dnase_model_boxplot_valid_corr.png', bbox_inches='tight')
plt.close()

plt.figure()
ax = sns.boxplot(data = df[[ 'train correlation dnn head', 'train correlation']], showmeans = True)
ax.set_xticklabels([f'DNN head model\n{df["train correlation dnn head"].mean(axis=0):.4f}', f'DNASE model\n{df["train correlation"].mean(axis=0):.4f}'])
plt.ylabel('Pearson Correlation Coefficient')
plt.title(f'Train set correlation for DNASE tracks')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnase_model/dnase_model_boxplot_train_corr.png', bbox_inches='tight')
plt.close()

plt.figure()
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
plt.title(f'Test set correlation for DNASE tracks')
plt.xlabel(f'Test correlation DNN head model')
plt.ylabel(f'Test correlation DNASE model')
sns.scatterplot(data = df, x = 'test correlation dnn head', y = 'test correlation', color = 'k')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnase_model/dnase_model_scatterplot_test.png', bbox_inches='tight')
plt.close()

plt.figure()
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
plt.title(f'Valid set correlation for DNASE tracks')
plt.xlabel(f'Valid correlation DNN head model')
plt.ylabel(f'Valid correlation DNASE model')
sns.scatterplot(data = df, x = 'valid correlation dnn head', y = 'valid correlation', color = 'k')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnase_model/dnase_model_scatterplot_valid.png', bbox_inches='tight')
plt.close()

plt.figure()
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
plt.title(f'Train set correlation for DNASE tracks')
plt.xlabel(f'Train correlation DNN head model')
plt.ylabel(f'Train correlation DNASE model')
sns.scatterplot(data = df, x = 'train correlation dnn head', y = 'train correlation', color = 'k')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnase_model/dnase_model_scatterplot_train.png', bbox_inches='tight')
plt.close()

plt.figure()
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
plt.title(f'Test set correlation for DNASE tracks')
plt.xlabel(f'Test correlation Enformer model')
plt.ylabel(f'Test correlation DNASE model')
sns.scatterplot(data = df, x = 'test correlation enformer', y = 'test correlation', color = 'k')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnase_model/dnase_model_scatterplot_test_enf_dnase.png', bbox_inches='tight')
plt.close()

plt.figure()
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
plt.title(f'Test set correlation for DNASE tracks')
plt.xlabel(f'Test correlation Enformer model')
plt.ylabel(f'Test correlation DNN head model')
sns.scatterplot(data = df, x = 'test correlation enformer', y = 'test correlation dnn head', color = 'k', label = 'DNN head model')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnase_model/dnase_model_scatterplot_test_enf_dnnhead.png', bbox_inches='tight')
plt.close()

plt.figure()
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
plt.title(f'Test set correlation for DNASE tracks')
plt.xlabel(f'Test correlation Enformer model')
plt.ylabel(f'Test correlation DNN head model')
sns.scatterplot(data = df, x = 'test correlation enformer', y = 'test correlation dnn head', color = 'k', s = 20, label = 'DNN head model')
sns.scatterplot(data = df, x = 'test correlation enformer', y = 'test correlation', color = 'b', s = 20, label = 'DNASE model', alpha = 0.5)
plt.legend()
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnase_model/dnase_model_scatterplot_test_enf_dnnhead_dnase.png', bbox_inches='tight')
plt.close()


