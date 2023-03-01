from datetime import datetime
start = datetime.now()
import sys
print(f'{start} Start of Python script {sys.argv[0]}')
import os
from natsort import natsorted
import torch

### old output 
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
# torch_big = torch.empty(size=(2213, 896, 5313))
# print(torch_big.shape)
# for i in range(1, 2213+1):
#     t = torch.load(f'/exports/humgen/idenhond/data/Enformer_validation/tensors_validation/output_seq{i}.pt', map_location=torch.device('cpu'))
#     print(i)
#     torch_big[i-1] =  t
# print(f'shape of torch with all output validation: {torch_big.shape}')
# output_file = '/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_output/output_validation.pt'
# torch.save(torch_big, output_file)


### new output
subset = str(sys.argv[1])

# # output validation
# if subset == 'valid':
#     print(subset)
#     torch_big = torch.empty(size=(2213, 896, 3072))
#     print(torch_big.shape)
#     for i in range(1, 2213+1):
#         t = torch.load(f'/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_embeddings_newmodel/embeddings_seq{i}.pt', map_location=torch.device('cpu'))
#         print(i)
#         torch_big[i-1] =  t
#     print(f'shape of torch with all embeddings validation: {torch_big.shape}')
#     output_file = '/exports/humgen/idenhond/data/Enformer_validation/embeddings_validation_pretrainedmodel.pt'
#     torch.save(torch_big, output_file)

# # output test
# if subset == 'test':
#     print(subset)
#     torch_big = torch.empty(size=(1937, 896, 3072))
#     print(torch_big.shape)
#     for i in range(1, 1937+1):
#         t = torch.load(f'/exports/humgen/idenhond/data/Enformer_test/Enformer_test_embeddings_newmodel/embeddings_seq{i}.pt', map_location=torch.device('cpu'))
#         print(i)
#         torch_big[i-1] =  t
#     print(f'shape of torch with all embeddins test: {torch_big.shape}')
#     output_file = '/exports/humgen/idenhond/data/Enformer_test/embeddings_test_pretrainedmodel.pt'
#     torch.save(torch_big, output_file)


# embeddings train 
if subset == 'trainone':
    print(subset)
    torch_big = torch.empty(size=(8503, 896, 3072)) # 8503 entries
    print(torch_big.shape)
    for i in range(1, 8504): # 1 tm 8503
        t = torch.load(f'/exports/humgen/idenhond/data/Enformer_train/Enformer_train_embeddings_newmodel/embeddings_seq{i}.pt', map_location=torch.device('cpu'))
        torch_big[i-1] = t
    print(f'shape of torch with all embeddins test: {torch_big.shape}')
    output_file = '/exports/humgen/idenhond/data/Enformer_train/embeddings_train_pretrainedmodel_one.pt'
    torch.save(torch_big, output_file)
        
if subset == 'traintwo':
    print(subset)
    torch_big = torch.empty(size=(8503, 896, 3072))
    print(torch_big.shape)
    index = 0
    for i in range(8504, 17007):  # 8504 tm 17006
        t = torch.load(f'/exports/humgen/idenhond/data/Enformer_train/Enformer_train_embeddings_newmodel/embeddings_seq{i}.pt', map_location=torch.device('cpu'))
        torch_big[index] = t
        index += 1
    print(f'shape of torch with all embeddins test: {torch_big.shape}')
    output_file = '/exports/humgen/idenhond/data/Enformer_train/embeddings_train_pretrainedmodel_two.pt'
    torch.save(torch_big, output_file)

if subset == 'trainthree':
    print(subset)
    torch_big = torch.empty(size=(8503, 896, 3072))
    print(torch_big.shape)
    index = 0
    for i in range(17007, 25510): # 17007 tm 25509
        t = torch.load(f'/exports/humgen/idenhond/data/Enformer_train/Enformer_train_embeddings_newmodel/embeddings_seq{i}.pt', map_location=torch.device('cpu'))
        torch_big[index] = t
        index += 1
    print(f'shape of torch with all embeddins test: {torch_big.shape}')
    output_file = '/exports/humgen/idenhond/data/Enformer_train/embeddings_train_pretrainedmodel_three.pt'
    torch.save(torch_big, output_file)

# HET ZIJN ER 34021                
if subset == 'trainfour':
    print(subset)
    torch_big = torch.empty(size=(8512, 896, 3072)) # deze is langer zodatje alle sequences hebt
    print(torch_big.shape)
    index = 0
    for i in range(25510, 34021+1): # 25510 tm 34021 
        t = torch.load(f'/exports/humgen/idenhond/data/Enformer_train/Enformer_train_embeddings_newmodel/embeddings_seq{i}.pt', map_location=torch.device('cpu'))
        torch_big[index] = t
        index += 1
    print(f'shape of torch with all embeddins test: {torch_big.shape}')
    output_file = '/exports/humgen/idenhond/data/Enformer_train/embeddings_train_pretrainedmodel_four.pt'
    torch.save(torch_big, output_file)
