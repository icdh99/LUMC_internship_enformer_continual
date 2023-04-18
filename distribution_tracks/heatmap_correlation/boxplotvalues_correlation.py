import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import sys
import pandas as pd

nr = int(sys.argv[1])
print(f'nr: {nr}')

# track 1 new tracks vs all old tracks

target_folder_oldtracks = '/exports/humgen/idenhond/data/Enformer_train/Enformer_train_targets'
target_folder_newtracks = '/exports/humgen/idenhond/data/Enformer_test/Enformer_test_targets_newtracks2703'

matrix_track1 = np.zeros([5313])

# for each sequence, add the correlation between the new and each old track to the big matrix
for nr_x in range(1, 1937+1): 
    target_new = torch.load(f'{target_folder_newtracks}/targets_seq{nr_x}.pt', map_location=torch.device('cpu')).squeeze()
    target_old = torch.load(f'{target_folder_oldtracks}/targets_seq{nr_x}.pt', map_location=torch.device('cpu')).squeeze()

    # SELECT TRACK 0 - 18 --> 1-19
    target_new_track1 = target_new[:,nr]

    # correlation between track1new and each old track
    corr_list_track1 = np.corrcoef(target_new_track1, target_old, rowvar=False)[0][1:]
    # corr_list_track1 = corr_list_track1
    matrix_track1 = np.add(matrix_track1, corr_list_track1)

    print(nr_x)
    # if nr_x == 1: break

final_matrix_track1 = matrix_track1/nr_x
print(f'final matrix track {nr}:')
# print(final_matrix_track1[:10])
# print(final_matrix_track1.shape)

pd.DataFrame(final_matrix_track1).to_csv(f'matrix_newtrack{nr}.csv', header = None, index = None)
