from datetime import datetime
start = datetime.now()

import sys
import os
from natsort import natsorted
print(f'{start} Start of Python script {sys.argv[0]}')
folder = '/exports/humgen/idenhond/enformer_dev/enformer-pytorch/data/tensors_test'
import torch
# for file in sorted(os.listdir(folder)):
#     print(file)




x = torch.empty(size=(1937, 896, 3072))
print(x.shape)
# print(x[0])

for i in range(1, 1937+1):
    t = torch.load(f'/exports/humgen/idenhond/enformer_dev/enformer-pytorch/data/tensors_test/embeddings_seq{i}.pt', map_location=torch.device('cpu'))
    print(t.shape)
    x[i-1] =  t

print(x.shape)
# print(x[0])

torch.save(x, 'all_embeddings_test.pt')
exit()

import torch 
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device (set to GPU if available):', device)

tensor_output_test_seq1 = '/exports/humgen/idenhond/enformer_dev/enformer-pytorch/data/tensors_test/output_seq1.pt'
tensor_out = torch.load(tensor_output_test_seq1, map_location=torch.device('cpu'))

print(f'information about tensour output sequence 1')
print(f'shape: {tensor_out.shape}')
print(f'dtype: {tensor_out.dtype}')

tensor_output_test_seq2 = '/exports/humgen/idenhond/enformer_dev/enformer-pytorch/data/tensors_test/output_seq2.pt'
tensor_out2 = torch.load(tensor_output_test_seq1, map_location=torch.device('cpu'))

t = torch.stack((tensor_out, tensor_out2), dim = 0)
print(f'shape of stacked tensor: {t.shape}')
print(f'shape of tensor 1: {t[0].shape}')
print(f'shape of tensor 2: {t[1].shape}')
print(f'dtype: {tensor_out.dtype}')
print(t[0])



exit()
torch.save(t, 'seqs_2.pt')



# print(t[0][-1])


# t = torch.load('/exports/humgen/idenhond/enformer_dev/enformer-pytorch/data/test-sample.pt', map_location=torch.device('cpu'))
# print(t)
# print(t.keys())
# print(t['sequence'].shape)
# print(t['sequence'].dtype)
# print(t['sequence'].device)
# print(t['target'].shape)
# print(t['target'].dtype)
# print(t['target'].device)





