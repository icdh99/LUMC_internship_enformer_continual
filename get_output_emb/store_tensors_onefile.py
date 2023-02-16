from datetime import datetime
start = datetime.now()
import sys
print(f'{start} Start of Python script {sys.argv[0]}')
import os
from natsort import natsorted
import torch

# # embeddings test
# torch_big = torch.empty(size=(1937, 896, 3072))
# print(torch_big.shape)
# for i in range(1, 1937+1):
#     t = torch.load(f'/exports/humgen/idenhond/enformer_dev/enformer-pytorch/data/tensors_test/embeddings_seq{i}.pt', map_location=torch.device('cpu'))
#     print(i)
#     torch_big[i-1] =  t
# print(f'shape of torch with all embeddings test: {torch_big.shape}')
# output_file = '/exports/humgen/idenhond/enformer_dev/enformer-pytorch/data/tensors_test/embeddings_test.pt'
# torch.save(torch_big, output_file)

# # output test
# torch_big = torch.empty(size=(1937, 896, 5313))
# print(torch_big.shape)
# for i in range(1, 1937+1):
#     t = torch.load(f'/exports/humgen/idenhond/enformer_dev/enformer-pytorch/data/tensors_test/output_seq{i}.pt', map_location=torch.device('cpu'))
#     print(i)
#     torch_big[i-1] =  t
# print(f'shape of torch with all output test: {torch_big.shape}')
# output_file = '/exports/humgen/idenhond/enformer_dev/enformer-pytorch/data/tensors_test/output_test.pt'
# torch.save(torch_big, output_file)

# # embeddings validation
# torch_big = torch.empty(size=(2213, 896, 3072))
# print(torch_big.shape)
# for i in range(1, 2213+1):
#     t = torch.load(f'/exports/humgen/idenhond/enformer_dev/enformer-pytorch/data/tensors_validation/embeddings_seq{i}.pt', map_location=torch.device('cpu'))
#     print(i)
#     torch_big[i-1] =  t
# print(f'shape of torch with all embeddings validation: {torch_big.shape}')
# output_file = '/exports/humgen/idenhond/enformer_dev/enformer-pytorch/data/tensors_validation/embeddings_validation.pt'
# torch.save(torch_big, output_file)

# output validation
torch_big = torch.empty(size=(2213, 896, 5313))
print(torch_big.shape)
for i in range(1, 2213+1):
    t = torch.load(f'/exports/humgen/idenhond/data/Enformer_validation/tensors_validation/output_seq{i}.pt', map_location=torch.device('cpu'))
    print(i)
    torch_big[i-1] =  t
print(f'shape of torch with all output validation: {torch_big.shape}')
output_file = '/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_output/output_validation.pt'
torch.save(torch_big, output_file)