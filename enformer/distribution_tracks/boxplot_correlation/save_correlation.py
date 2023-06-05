import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import os
import sys

targets_or_outputs = sys.argv[1]
print(f'subset: {targets_or_outputs}')

if targets_or_outputs == 'targets':
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

    np.savetxt('final_correlation_matrix_alltracks_testset_targets.csv', final_matrix, delimiter=',')

if targets_or_outputs == 'outputs':
## all tracks test set outputs
    output_folder_oldtracks = '/exports/archive/hg-funcgenom-research/idenhond/Enformer_test/Enformer_test_output_dnnhead_retrain2703'
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

    np.savetxt('final_correlation_matrix_alltracks_testset_outputs.csv', final_matrix, delimiter=',')


if targets_or_outputs == 'outputs_enformer':
## all tracks test set outputs from enformer model (pytorch model)
    output_folder = '/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output'

    corr_matrix = np.zeros([5313, 5313])
    np.set_printoptions(linewidth=400)   
    for nr_x in range(1, 1937+1):      
        output = torch.load(f'{output_folder}/output_seq{nr_x}.pt', map_location=torch.device('cpu')).squeeze()
        x = np.corrcoef(output, rowvar=False)
        corr_matrix = np.add(corr_matrix, x)
        # if nr_x == 50: break

    final_matrix = corr_matrix/nr_x
    print(final_matrix.shape)

    np.savetxt('final_correlation_matrix_alltracks_testset_outputs_enformer.csv', final_matrix, delimiter=',')

if targets_or_outputs == 'targets_atac':
## all tracks test set outputs from atac model

    df = pd.read_csv('/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_classes_counts.csv', sep = ',')
    order_list = df['Index old'].to_list()
    order_list = list(set(order_list))

    target_folder = '/exports/humgen/idenhond/data/Enformer_test/Human_ATAC_test_targets'

    corr_matrix = np.zeros([66, 66])
    np.set_printoptions(linewidth=400)   
    for nr_x in range(1, 1937+1):      
        target = torch.load(f'{target_folder}/targets_seq{nr_x}.pt', map_location=torch.device('cpu')).squeeze()
        target = target[:, order_list]
        x = np.corrcoef(target, rowvar=False)
        corr_matrix = np.add(corr_matrix, x)
        # if nr_x == 50: break

    final_matrix = corr_matrix/nr_x
    print(final_matrix.shape)

    np.savetxt('final_correlation_matrix_testset_atac_targets_ordered.csv', final_matrix, delimiter=',')

if targets_or_outputs == 'outputs_atac':
## all tracks test set outputs from atac model
    # order_list = 
    df = pd.read_csv('/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_classes_counts.csv', sep = ',')
    order_list = df['Index old'].to_list()
    order_list = list(set(order_list))

    output_folder = '/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output_humanatac' 

    corr_matrix = np.zeros([66, 66])
    np.set_printoptions(linewidth=400)   

    for nr_x in range(1, 1937+1):      
        output = torch.load(f'{output_folder}/output_seq{nr_x}.pt', map_location=torch.device('cpu')).squeeze()

        output = output[:, order_list]
        x = np.corrcoef(output, rowvar=False)
        corr_matrix = np.add(corr_matrix, x)
        # if nr_x == 50: break

    final_matrix = corr_matrix/nr_x
    print(final_matrix.shape)

    np.savetxt('final_correlation_matrix_testset_atac_outputs_ordered.csv', final_matrix, delimiter=',')