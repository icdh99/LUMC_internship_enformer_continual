import torch
from torchmetrics import PearsonCorrCoef
import matplotlib.pyplot as plt
import seaborn as sns
import os
from natsort import natsorted
import numpy as np 

"""
dnn model new tracks only

test set

predictions: /exports/humgen/idenhond/data/Enformer_test/Enformer_test_output_dnnhead_newtracks
1 file = output_seq1.pt
torch.Size([1, 896, 19])

targets: /exports/humgen/idenhond/data/Enformer_test/Enformer_test_targets_newtracks2703
1 file = targets_seq1.pt
torch.Size([1, 896, 19])
"""

pred_folder = '/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output_dnnhead_newtracks'
target_folder = '/exports/humgen/idenhond/data/Enformer_test/Enformer_test_targets_newtracks2703'

# pearson = PearsonCorrCoef(num_outputs=19)

# corr_pertrack = np.zeros(19)
# print(corr_pertrack)

i = 0 
for file_pred, file_target in zip(natsorted(os.listdir(pred_folder)), natsorted(os.listdir(target_folder))):
    print(file_pred, file_target)
    p = torch.load(f'{pred_folder}/{file_pred}', map_location = torch.device('cpu')).squeeze()
    t = torch.load(f'{target_folder}/{file_target}', map_location = torch.device('cpu')).squeeze()
    print(p.shape, t.shape)

    # figure, axis = plt.subplots(nrows = 4, ncols = 5)
    
    # axis[0,0].plot(t[:0])
    # axis[0,1].plot(t[:0])
    # axis[0,2].plot(t[:0])
    # axis[0,3].plot(t[:0])
    # axis[0,4].plot(t[:0])

    # axis[1,0].plot(t[:0])
    # axis[1,1].plot(t[:0])
    # axis[1,2].plot(t[:0])
    # axis[1,3].plot(t[:0])
    # axis[1,4].plot(t[:0])
    

    plt.plot(t[:,0], linewidth = 0.7, label = 'target')
    plt.plot(p[:,0], linewidth = 1, label = 'predicted')
    plt.title(f'seq {i} track 0')
    plt.legend()
    plt.savefig(f'seq{i}_track0.png')
    plt.close()

    plt.plot(t[:,1], linewidth = 0.7, label = 'target')
    plt.plot(p[:,1], linewidth = 1, label = 'predicted')
    plt.plot(p[:,1]*2, linewidth = 1, label = 'predicted *2')
    plt.title(f'seq {i} track 1')
    plt.legend()
    plt.savefig(f'seq{i}_track1.png')
    plt.close()

    plt.plot(t[:,3], linewidth = 0.7, label = 'target')
    plt.plot(p[:,3], linewidth = 1, label = 'predicted')
    plt.title(f'seq {i} track 3')
    plt.legend()
    plt.savefig(f'seq{i}_track3.png')
    plt.close()


    # a = pearson(p, t)
    # print(a)
    # print(a.numpy())

    i += 1
    # corr_pertrack
    # f = np.add(corr_pertrack, a.numpy())
    if i == 10: break
# print(f)

    


