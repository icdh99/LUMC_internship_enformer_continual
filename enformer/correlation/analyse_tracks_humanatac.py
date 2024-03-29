import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_file = '/exports/humgen/idenhond/data/basenji_preprocess/output_tfr_human_atac/targets.txt'
df = pd.read_csv(input_file, sep = '\t')
print(f'Number of tracks: {df.shape[0]}')

df_correlation_test = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_humanatac.csv', index_col = 0, names = ['correlation test']).tail(-1)
df_correlation_valid = pd.read_csv('//exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_humanatac.csv', index_col = 0, names = ['correlation valid']).tail(-1)

df['test correlation'] = df_correlation_test['correlation test']
df['valid correlation'] = df_correlation_valid['correlation valid']


print(f'mean correlation score test: {df["test correlation"].mean(axis=0):.4f}')
print(f'mean correlation score valid: {df["valid correlation"].mean(axis=0):.4f}')

print(df)

df = df.sort_values(by=['test correlation'], ascending=False)
print(df)

plt.figure()
ax = sns.boxplot(data = df[[ 'test correlation', 'valid correlation']], showmeans = True)
ax.set_xticklabels([f'Test set\n{df["test correlation"].mean(axis=0):.4f}', f'Validation set\n{df["valid correlation"].mean(axis=0):.4f}'])
plt.ylabel('Pearson Correlation Coefficient')
plt.title(f'Correlation for 66 human ATAC-seq tracks')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/human_atac/human_atac_boxplot_test_val.png', bbox_inches='tight')
plt.close()

plt.figure()
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df, x = 'test correlation', y = 'valid correlation', c = 'k')
plt.title(f'Pearson Correlation Coefficient for 66 human ATAC-seq tracks')
plt.xlabel(f'Test set')
plt.ylabel('Validation set')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/human_atac/human_atac_scatterplot_test_val.png', bbox_inches='tight')
plt.close()

PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'},
    'meanprops': {'markerfacecolor':'black', 'markeredgecolor':'black'}
}
# meanprops={"markerfacecolor":"black", "markeredgecolor":"black"} # "marker":"^", 
 
plt.figure(figsize=(3.8, 1.4)) 
ax = sns.boxplot(data = df[[ 'test correlation']], showmeans = True, orient = 'h', width = 0.8, **PROPS)
# ax.set_xticklabels([f'Test set\n{df["test correlation"].mean(axis=0):.4f}'])
plt.ylabel(f'Test set\n{df["test correlation"].mean(axis=0):.4f}')
plt.xlabel('Pearson Correlation Coefficient')
# plt.title(f'Correlation for 66 human ATAC-seq tracks')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/human_atac/human_atac_boxplot_test.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 2)) 
sns.set_context("poster")
sns.swarmplot(data = df, x = 'test correlation', color = 'black', s = 10)
plt.ylabel(f'Test set')
plt.xlabel('Pearson correlation coefficient')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/human_atac/human_atac_swarmplot_test_poster.png', bbox_inches='tight')
plt.close()

plt.figure(figsize=(10, 2)) 
sns.set_context("talk")
sns.swarmplot(data = df, x = 'test correlation', color = 'black')
plt.ylabel(f'Test set')
plt.xlabel('Pearson correlation coefficient')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/human_atac/human_atac_swarmplot_test_talk.png', bbox_inches='tight')
plt.close()