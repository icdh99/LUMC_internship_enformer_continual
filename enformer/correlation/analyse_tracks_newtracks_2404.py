import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_file = '/exports/humgen/idenhond/data/basenji_preprocess/output_tfr_27newtracks_human/targets.txt'
df = pd.read_csv(input_file, sep = '\t')
print(f'Number of tracks: {df.shape[0]}')


df_correlation_test = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_newtracks_2404.csv', index_col = 0, names = ['correlation test']).tail(-1)
df_correlation_valid = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_newtracks_2404.csv', index_col = 0, names = ['correlation valid']).tail(-1)
df_correlation_train = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train_newtracks_2404.csv', index_col = 0, names = ['correlation train']).tail(-1)

df['test correlation'] = df_correlation_test['correlation test']
df['valid correlation'] = df_correlation_valid['correlation valid']
df['train correlation'] = df_correlation_train['correlation train']
print(df)

print(f'mean correlation score test: {df["test correlation"].mean(axis=0):.4f}')
print(f'mean correlation score valid: {df["valid correlation"].mean(axis=0):.4f}')
print(f'mean correlation score train: {df["train correlation"].mean(axis=0):.4f}')

plt.figure()
ax = sns.boxplot(data = df[[ 'test correlation', 'valid correlation', 'train correlation']], showmeans = True)
ax.set_xticklabels([f'Test set\n{df["test correlation"].mean(axis=0):.4f}', f'Validation set\n{df["valid correlation"].mean(axis=0):.4f}', f'Train set\n{df["train correlation"].mean(axis=0):.4f}'])
plt.ylabel('Pearson Correlation Coefficient')
plt.title(f'Correlation for 27 new human tracks')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/newtracks_2404/newtracks_2404_boxplot.png', bbox_inches='tight')
plt.close()

plt.figure()
ax = sns.boxplot(data = df, y = 'test correlation', x = 'assay type', showmeans = True)
# ax.set_xticklabels([f'Test set\n{df["test correlation"].mean(axis=0):.4f}', f'Validation set\n{df["valid correlation"].mean(axis=0):.4f}', f'Train set\n{df["train correlation"].mean(axis=0):.4f}'])
plt.ylabel('Pearson Correlation Coefficient')
plt.xlabel('Assay type')
plt.title(f'Correlation for 27 new human tracks per assay type - Test set')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/newtracks_2404/newtracks_2404_boxplot_test_perassaytype.png', bbox_inches='tight')
plt.close()

plt.figure()
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df, x = 'test correlation', y = 'valid correlation', c = 'k')
plt.title(f'Pearson Correlation Coefficient for 27 new human tracks')
plt.xlabel(f'Test set')
plt.ylabel('Validation set')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/newtracks_2404/newtracks_2404_scatterplot_testval.png', bbox_inches='tight')
plt.close()

plt.figure()
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df, x = 'test correlation', y = 'valid correlation', hue = 'assay type')
plt.title(f'Pearson Correlation Coefficient for 27 new human tracks')
plt.xlabel(f'Test set')
plt.ylabel('Validation set')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/newtracks_2404/newtracks_2404_scatterplot_testval_assaytype.png', bbox_inches='tight')
plt.close()

plt.figure()
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df, x = 'test correlation', y = 'train correlation', c = 'k')
plt.title(f'Pearson Correlation Coefficient for 27 new human tracks')
plt.xlabel(f'Test set')
plt.ylabel('Train set')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/newtracks_2404/newtracks_2404_scatterplot_testtrain.png', bbox_inches='tight')
plt.close()