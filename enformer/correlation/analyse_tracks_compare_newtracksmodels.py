'''
this script compares the performance of the two models trained on new human tracks
model 1: train_newtracks_2703 (22 tracks) inlcuding 3 enformer tracks
model 2: train_newtrakcs_2404 (27 tracks)
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_file = '/exports/humgen/idenhond/data/basenji_preprocess/output_tfr_27newtracks_human/targets.txt'
df_new = pd.read_csv(input_file, sep = '\t', index_col='index')
print(f'Number of tracks: {df_new.shape[0]}')
df_correlation_test = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_newtracks_2404.csv', index_col = 0, names = ['correlation test']).tail(-1)
df_new['test correlation'] = df_correlation_test['correlation test']
print(df_new.head())
print(df_new.columns)

input_file = '/exports/humgen/idenhond/data/basenji_preprocess/output_tfr/targets.txt'
df_old = pd.read_csv(input_file, sep = '\t')
print(f'Number of tracks: {df_old.shape[0]}')
df_old = df_old.drop(labels = [3,4,5], axis = 'index').reset_index(drop=True)
df_correlation_test = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_dnn_head_newtracks.csv', index_col = 0, names = ['correlation test']).tail(-1)
df_old['test correlation'] = df_correlation_test['correlation test']
print(df_old.head())
print(df_old.columns)

print(df_new[['identifier', 'test correlation']])
print(df_old[['index', 'test correlation']])


new = list(df_new['identifier'])
old = list(df_old['index'])
print( list(set(new) & set(old)))
print(len(list(set(new) & set(old))))

df_merged = df_new.merge(df_old, left_on='identifier', right_on='index', suffixes = ['_old', '_new'])
print(df_merged)
print(df_merged.columns)
print(df_merged[['identifier_old', 'test correlation_old', 'test correlation_new']])

plt.figure()
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df_merged, x = 'test correlation_old', y = 'test correlation_new', c = 'k', hue = 'assay type')
plt.title(f'Pearson Correlation Coefficient for new vs old new tracks')
plt.xlabel(f'Test set old')
plt.ylabel('Test set new')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/newtracks_oldvsnew_scatterplot_test.png', bbox_inches='tight')
plt.close()

plt.figure()
ax = sns.boxplot(data = df_merged[[ 'test correlation_old', 'test correlation_new']], showmeans = True)
ax.set_xticklabels([f'Test set old\n{df_merged["test correlation_old"].mean(axis=0):.4f}', f'Test set new\n{df_merged["test correlation_new"].mean(axis=0):.4f}'])
plt.ylabel('Pearson Correlation Coefficient')
plt.title(f'Correlation for old and new new human tracks')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/newtracks_oldvsnew_boxplot_test.png', bbox_inches='tight')
plt.close()