from datetime import datetime
start = datetime.now()
import sys
print(f'{start} Start of Python script {sys.argv[0]}')
import torch

"""
Information about CPU & GPU usage from PyTorch
"""

# get index of currently selected device
print(f'Index of currently selected device: {torch.cuda.current_device()}') # returns 0 in my case
# get number of GPUs available
print(f'Number of GPUs available: {torch.cuda.device_count()}') # returns 1 in my case
# get the name of the device
print(f'Name of the device: {torch.cuda.get_device_name(0)}') 
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device (set to GPU if available):', device)
# print(f'torch.cuda_is_available(): {torch.cuda.is_available()}\n')
#Additional Info when using cuda
if device.type == 'cuda':
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print('Memory Usage:', 'Allocated', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB,', 'Cached', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB\n')
    # print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    # print('Cached:', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')   # torch.cuda.memory_cached has been renamed to torch.cuda.memory_reserved

"""
Test Enformer model
for 1 sequence (seq = torch.randint(0, 5, (1, 196_608))) the script takes 2 minutes and 15GB memory
- with return embeddings = True
2 seq = 30GB
3 seq = 36GB
4 seq = 52GB
5 seq = 70GB
"""

from enformer_pytorch import Enformer, seq_indices_to_one_hot

model = Enformer.from_hparams(
    dim = 1536,
    depth = 11,
    heads = 8,
    output_heads = dict(human = 5313, mouse = 1643),
    target_length = 896,
)

# enformer = Enformer.from_pretrained('EleutherAI/enformer-official-rough')

seq = torch.randint(0, 5, (1, 196_608)) # indicate number of sequences X with size argument (X, 196_608)

print(f'type seq: {type(seq)}')
print(f"Shape of tensor seq: {seq.shape}")
print(f"Datatype of tensor seq: {seq.dtype}")
print(f"Device tensor seq is stored on: {seq.device}")



output, embeddings = model(seq, return_embeddings = True, head = 'human')
print(f'type output: {type(output)}')
print(f'type embeddings: {type(embeddings)}')


print('output information')
print(f"Shape of tensor output: {output.shape}")
print(f"Datatype of tensor output: {output.dtype}")
print(f"Device tensor output is stored on: {output.device}")
print(type(output)) # (1, 896, 5313)
# print(type(output['mouse'])) # (1, 896, 1643)
print((output).shape) # (1, 896, 5313)
# print((output['mouse']).shape) # (1, 896, 1643)

print('embedding information')
print(f"Shape of tensor embeddings: {embeddings.shape}")
print(f"Datatype of tensor embeddings: {embeddings.dtype}")
print(f"Device tensor embeddings is stored on: {embeddings.device}")

print(f'Time: {datetime.now() - start}') 