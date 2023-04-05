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

target_folder_oldtracks = '/exports/humgen/idenhond/data/Enformer_train/Enformer_train_targets'
target_folder_newtracks = '/exports/humgen/idenhond/data/Enformer_test/Enformer_test_targets_newtracks2703'

corr_matrix = np.zeros([5332, 5332])
np.set_printoptions(linewidth=400)   

for nr_x in range(1, 1937+1):      
    print(nr_x)
    target_new = torch.load(f'{target_folder_newtracks}/targets_seq{nr_x}.pt', map_location=torch.device('cpu')).squeeze()
    target_old = torch.load(f'{target_folder_oldtracks}/targets_seq{nr_x}.pt', map_location=torch.device('cpu')).squeeze()
    target = torch.cat((target_old, target_new), dim = 1)
    # print(f'shape of x sequence {nr_x}: {target.shape}')
    x = np.corrcoef(target, rowvar=False)
    # print(x.shape)
    corr_matrix = np.add(corr_matrix, x)
    # print(x)
    # if nr_x == 1: break

final_matrix = corr_matrix/nr_x
print(final_matrix)


plt.subplots(figsize=(50,50))
plt.imshow(final_matrix, cmap='hot')
plt.colorbar(shrink=0.5)
plt.savefig('heatmap_alltrackspng')
exit()
# pd.DataFrame(final_matrix).to_csv('heatmap_testset_newtracks.csv')

f, ax = plt.subplots(figsize = (11,9))
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(final_matrix, vmin = 0.0, vmax = 1.0, cmap=cmap, square=True, linewidths=0.5, cbar_kws={"shrink": .5})
plt.savefig('heatmap_testset_1trackalltracks.png', dpi = 600)

