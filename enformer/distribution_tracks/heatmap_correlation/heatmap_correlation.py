import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os
import sys

subset = sys.argv[1]
print(f'subset: {subset}')

if subset == 'test':
    ## all tracks test set targets
    target_folder_oldtracks = '/exports/humgen/idenhond/data/Enformer_test/Enformer_test_targets_perseq'
    target_folder_newtracks = '/exports/humgen/idenhond/data/Enformer_test/Newtracks_2404_test_targets'

    corr_matrix = np.zeros([5340, 5340])
    np.set_printoptions(linewidth=400)   
    for nr_x in range(1, 1937+1):      
        print(nr_x)
        target_old = torch.load(f'{target_folder_oldtracks}/targets_seq{nr_x}.pt', map_location=torch.device('cpu')).squeeze()
        target_new = torch.load(f'{target_folder_newtracks}/targets_seq{nr_x}.pt', map_location=torch.device('cpu')).squeeze()
        target = torch.cat((target_old, target_new), dim = 1)
        x = np.corrcoef(target, rowvar=False)
        corr_matrix = np.add(corr_matrix, x)
        # if nr_x == 50: break

    final_matrix = corr_matrix/nr_x

    plt.subplots(figsize=(50,50))
    cmap = sns.color_palette("Blues", as_cmap=True)
    plt.imshow(final_matrix, cmap=cmap)
    plt.colorbar(shrink=0.5)
    plt.savefig('heatmap_alltracks_testset_targets.png')

    ## all tracks test set outputs
    output_folder_oldtracks = '/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output_dnnhead_retrain2703'
    output_folder_newtracks = '/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output_newtracks_2404'

    corr_matrix = np.zeros([5340, 5340])
    np.set_printoptions(linewidth=400)   
    for nr_x in range(1, 1937+1):      
        print(nr_x)
        output_old = torch.load(f'{output_folder_oldtracks}/output_seq{nr_x}.pt', map_location=torch.device('cpu')).squeeze()
        output_new = torch.load(f'{output_folder_newtracks}/output_seq{nr_x}.pt', map_location=torch.device('cpu')).squeeze()
        output = torch.cat((output_old, output_new), dim = 1)
        x = np.corrcoef(output, rowvar=False)
        corr_matrix = np.add(corr_matrix, x)
        # if nr_x == 50: break
    final_matrix = corr_matrix/nr_x

    plt.subplots(figsize=(50,50))
    cmap = sns.color_palette("Blues", as_cmap=True)
    plt.imshow(final_matrix, cmap=cmap)
    plt.colorbar(shrink=0.5)
    plt.savefig('heatmap_alltracks_testset_output.png')
















exit()
if subset == 'train_targets':
    ## te groot
    ## all tracks train set targets
    target_folder_oldtracks = '/exports/humgen/idenhond/data/Enformer_train/Enformer_train_targets'
    target_folder_newtracks = '/exports/humgen/idenhond/data/Enformer_train/Newtracks_2404_train_targets'

    corr_matrix = np.zeros([5340, 5340])
    np.set_printoptions(linewidth=400)   
    for nr_x in range(1, 34021+1):      
        print(nr_x)
        target_old = torch.load(f'{target_folder_oldtracks}/targets_seq{nr_x}.pt', map_location=torch.device('cpu')).squeeze()
        target_new = torch.load(f'{target_folder_newtracks}/targets_seq{nr_x}.pt', map_location=torch.device('cpu')).squeeze()
        target = torch.cat((target_old, target_new), dim = 1)
        x = np.corrcoef(target, rowvar=False)
        corr_matrix = np.add(corr_matrix, x)
        # if nr_x == 10: break

    final_matrix = corr_matrix/nr_x

    plt.subplots(figsize=(50,50))
    cmap = sns.color_palette("Blues", as_cmap=True)
    plt.imshow(final_matrix, cmap=cmap)
    plt.colorbar(shrink=0.5)
    plt.savefig('heatmap_alltracks_trainset_targets.png')

if subset == 'train_outputs':
    ## te groot
    ## all tracks train set outputs
    output_folder_oldtracks = '/exports/archive/hg-funcgenom-research/idenhond/Enformer_train/Enformer_train_output_dnnhead_retrain2703'
    output_folder_newtracks = '/exports/humgen/idenhond/data/Enformer_train/Enformer_train_output_newtracks_2404'

    corr_matrix = np.zeros([5340, 5340])
    np.set_printoptions(linewidth=400)   
    for nr_x in range(1, 34021+1):      
        print(nr_x)
        output_old = torch.load(f'{output_folder_oldtracks}/output_seq{nr_x}.pt', map_location=torch.device('cpu')).squeeze()
        output_new = torch.load(f'{output_folder_newtracks}/output_seq{nr_x}.pt', map_location=torch.device('cpu')).squeeze()
        output = torch.cat((output_old, output_new), dim = 1)
        x = np.corrcoef(output, rowvar=False)
        corr_matrix = np.add(corr_matrix, x)
        # if nr_x == 10: break
    final_matrix = corr_matrix/nr_x

    plt.subplots(figsize=(50,50))
    cmap = sns.color_palette("Blues", as_cmap=True)
    plt.imshow(final_matrix, cmap=cmap)
    plt.colorbar(shrink=0.5)
    plt.savefig('heatmap_alltracks_trainset_output.png')

