import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os

'''
test set
1937 sequences

5313 + 19 = 5332 tracks
'''

target_folder_newtracks = '/exports/humgen/idenhond/data/Enformer_test/Enformer_test_targets_newtracks2703'

corr_matrix = np.zeros([19, 19])

for nr_x in range(1, 1937+1):      
    print(nr_x)
    target = torch.load(f'{target_folder_newtracks}/targets_seq{nr_x}.pt', map_location=torch.device('cpu')).squeeze()
    print(f'shape of x sequence {nr_x}: {target.shape}')
    x = np.corrcoef(target, rowvar=False)
    # print(x.shape)
    corr_matrix = np.add(corr_matrix, x)
    # print(x)
    # if nr_x == 1: break

final_matrix = corr_matrix/nr_x
print(final_matrix)

f, ax = plt.subplots(figsize = (11,9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(final_matrix, vmin = 0.0, vmax = 1.0, cmap=cmap, square=True, linewidths=0.5, cbar_kws={"shrink": .5})
plt.savefig('heatmap_newtracks.png', dpi = 600)

