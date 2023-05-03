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
df = pd.read_csv('/exports/humgen/idenhond/data/basenji_preprocess/output_tfr_27newtracks_human/targets.txt', sep = '\t')
print(df)
names = list(df['assay type'])
names = list(df['description'])
# names = [' '.join(name.split('_')[2:]) for name in names]

## correlation matrix for test targets
corr_matrix = np.zeros([27, 27])
np.set_printoptions(linewidth=400)   

for nr_x in range(1, 1937+1):      
    target = torch.load(f'/exports/humgen/idenhond/data/Enformer_test/Newtracks_2404_test_targets/targets_seq{nr_x}.pt', map_location=torch.device('cpu')).squeeze()
    x = np.corrcoef(target, rowvar=False)
    corr_matrix = np.add(corr_matrix, x)

final_matrix = corr_matrix/nr_x
print(final_matrix[0])

xticks = [*range(0, 27)]
yticks = [*range(26, -1,  -1)]
matrix = np.triu(final_matrix)
np.fill_diagonal(matrix, False)
f, ax = plt.subplots(figsize = (11,9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
cmap = sns.color_palette("Blues", as_cmap=True)
p = sns.heatmap(final_matrix, vmin = 0.0, vmax = 1.0, cmap=cmap, square=True, mask = matrix, linewidths=0.5, cbar_kws={"shrink": .5})
plt.xticks(np.arange(0.5, 27), names, rotation = 90, fontsize = 9)
plt.yticks(np.arange(0.5, 27), names, rotation = 0, fontsize = 9)
plt.savefig('heatmap_27newtracks_testset_targets.png', bbox_inches = 'tight')

## correlation matrix for test predictions
corr_matrix = np.zeros([27, 27])
np.set_printoptions(linewidth=400)   

for nr_x in range(1, 1937+1):      
    target = torch.load(f'/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output_newtracks_2404/output_seq{nr_x}.pt', map_location=torch.device('cpu')).squeeze()
    x = np.corrcoef(target, rowvar=False)
    corr_matrix = np.add(corr_matrix, x)

final_matrix = corr_matrix/nr_x
print(final_matrix[0])

xticks = [*range(0, 27)]
yticks = [*range(26, -1,  -1)]
matrix = np.triu(final_matrix)
np.fill_diagonal(matrix, False)
f, ax = plt.subplots(figsize = (11,9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
cmap = sns.color_palette("Blues", as_cmap=True)
p = sns.heatmap(final_matrix, vmin = 0.0, vmax = 1.0, cmap=cmap, square=True, mask = matrix, linewidths=0.5, cbar_kws={"shrink": .5})
plt.xticks(np.arange(0.5, 27), names, rotation = 90, fontsize = 9)
plt.yticks(np.arange(0.5, 27), names, rotation = 0, fontsize = 9)
plt.savefig('heatmap_27newtracks_testset_output.png', bbox_inches = 'tight')

## correlation matrix for test predictions (new model with more layers 02/05)
corr_matrix = np.zeros([27, 27])
np.set_printoptions(linewidth=400)   

for nr_x in range(1, 1937+1):      
    target = torch.load(f'/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output_newtracks_newmodel_0205/output_seq{nr_x}.pt', map_location=torch.device('cpu')).squeeze()
    x = np.corrcoef(target, rowvar=False)
    corr_matrix = np.add(corr_matrix, x)

final_matrix = corr_matrix/nr_x
print(final_matrix[0])

xticks = [*range(0, 27)]
yticks = [*range(26, -1,  -1)]
matrix = np.triu(final_matrix)
np.fill_diagonal(matrix, False)
f, ax = plt.subplots(figsize = (11,9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
cmap = sns.color_palette("Blues", as_cmap=True)
p = sns.heatmap(final_matrix, vmin = 0.0, vmax = 1.0, cmap=cmap, square=True, mask = matrix, linewidths=0.5, cbar_kws={"shrink": .5})
plt.xticks(np.arange(0.5, 27), names, rotation = 90, fontsize = 9)
plt.yticks(np.arange(0.5, 27), names, rotation = 0, fontsize = 9)
plt.savefig('heatmap_27newtracks_newmodel_0205_testset_output.png', bbox_inches = 'tight')