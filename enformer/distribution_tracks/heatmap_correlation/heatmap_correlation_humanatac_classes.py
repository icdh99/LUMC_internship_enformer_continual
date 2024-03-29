import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os

'''
test set
1937 sequences
'''

## labels
df_classes = pd.read_csv('/exports/humgen/idenhond/targets_classes.csv', index_col='index', sep = ';')
cols = ['Class', 'Subclass', 'Ac-level annotation']
df_classes['names'] = df_classes[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
names = list(df_classes['names'])

new_indices = df_classes['Index old']

## correlation matrix for test targets
corr_matrix = np.zeros([66, 66])
np.set_printoptions(linewidth=400)   

for nr_x in range(1, 1937+1):      
    target = torch.load(f'/exports/humgen/idenhond/data/Enformer_test/Human_ATAC_test_targets/targets_seq{nr_x}.pt', map_location=torch.device('cpu')).squeeze()
    x = np.corrcoef(target, rowvar=False)
    corr_matrix = np.add(corr_matrix, x)

final_matrix = corr_matrix/nr_x
final_matrix = final_matrix[new_indices, :]
final_matrix = final_matrix[:, new_indices]

xticks = [*range(0, 66)]
yticks = [*range(65, -1,  -1)]
matrix = np.triu(final_matrix)
np.fill_diagonal(matrix, False)
f, ax = plt.subplots(figsize = (11,9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
cmap = sns.color_palette("Blues", as_cmap=True)
p = sns.heatmap(final_matrix, vmin = 0.0, vmax = 1.0, cmap=cmap, square=True, mask = matrix, linewidths=0.5, cbar_kws={"shrink": .5})
plt.xticks(np.arange(0.5, 66), names, rotation = 90, fontsize = 5)
plt.yticks(np.arange(0.5, 66), names, rotation = 0, fontsize = 5)
plt.savefig('heatmap_humanatac_classes_testset_targets.png', bbox_inches = 'tight')

## correlation matrix for test predictions
corr_matrix = np.zeros([66, 66])
np.set_printoptions(linewidth=400)   

for nr_x in range(1, 1937+1):      
    target = torch.load(f'/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output_humanatac/output_seq{nr_x}.pt', map_location=torch.device('cpu')).squeeze()
    x = np.corrcoef(target, rowvar=False)
    corr_matrix = np.add(corr_matrix, x)

final_matrix = corr_matrix/nr_x
final_matrix = final_matrix[new_indices, :]
final_matrix = final_matrix[:, new_indices]

xticks = [*range(0, 66)]
yticks = [*range(65, -1,  -1)]
matrix = np.triu(final_matrix)
np.fill_diagonal(matrix, False)
f, ax = plt.subplots(figsize = (11,9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
cmap = sns.color_palette("Blues", as_cmap=True)
p = sns.heatmap(final_matrix, vmin = 0.0, vmax = 1.0, cmap=cmap, square=True, mask = matrix, linewidths=0.5, cbar_kws={"shrink": .5})
plt.xticks(np.arange(0.5, 66), names, rotation = 90, fontsize = 5)
plt.yticks(np.arange(0.5, 66), names, rotation = 0, fontsize = 5)
plt.savefig('heatmap_humanatac_classes_testset_output.png', bbox_inches = 'tight')