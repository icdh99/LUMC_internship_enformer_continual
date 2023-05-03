from datetime import datetime
start = datetime.now()
import sys
print(f'{start} Start of Python script {sys.argv[0]}')
import torch

"""
Information about CPU & GPU usage from PyTorch
"""
# get index of currently selected device
# print(f'Index of currently selected device: {torch.cuda.current_device()}') # returns 0 in my case
# get number of GPUs available
# print(f'Number of GPUs available: {torch.cuda.device_count()}') # returns 1 in my case
# get the name of the device
# print(f'Name of the device: {torch.cuda.get_device_name(0)}') 
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device (set to GPU if available):', device)
# print(f'torch.cuda_is_available(): {torch.cuda.is_available()}\n')
#Additional Info when using cuda
if device.type == 'cuda':
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print('Memory Usage:', 'Allocated', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB,', 'Cached', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB\n')

"""
Test Enformer model
input sequence derived from fasta file
"""
from enformer_pytorch import Enformer, seq_indices_to_one_hot, GenomeIntervalDataset
import polars as pl

# load sequences
filter_train = lambda df: df.filter(pl.col('column_4') == 'valid')        
ds = GenomeIntervalDataset(
    bed_file = '/exports/humgen/idenhond/Enformer_data_google/data_human_sequences.bed',
    fasta_file = '/exports/humgen/idenhond/genomes/hg38.ml.fa',                    # path to fasta file
    filter_df_fn = filter_train,                        # filter dataframe function --> nu niet nodig
    return_seq_indices = True,                          # return nucleotide indices (ACGTN) or one hot encodings
    shift_augs = (-2, 2),                               # random shift augmentations from -2 to +2 basepairs --> even checken wat dit doet
    context_length = 196_608,
    # this can be longer than the interval designated in the .bed file,
    # in which case it will take care of lengthening the interval on either sides
    # as well as proper padding if at the end of the chromosomes
    chr_bed_to_fasta_map = {} #'chr1': 'chromosome1'
)
print(f'number of sequences in ds object: {len(ds)}')

# load model
model = Enformer.from_hparams(
    dim = 1536,
    depth = 11,
    heads = 8,
    output_heads = dict(human = 5313, mouse = 1643),
    target_length = 896 
)

# number of sequences to generate model output for
length = 2
# length = len(ds) 

# tensor to store embeddings and outptus in 
embeddings_tensor = torch.empty(size = (length, 896, 3072))
output_tensor = torch.empty(size = (length, 896, 5313))

for i in range(length):
    seq = ds[i]
    print(i)

    output, embeddings = model(seq, return_embeddings = True, head = 'human')
    print('seq information')
    print(f'type seq: {type(seq)}')
    print(f"Shape seq: {seq.shape}")
    print(f"Datatype seq: {seq.dtype}") #int64
    print(f"Device seq is stored on: {seq.device}\n")

    print(f'output information')
    print(f'type output: {type(output)}')
    print(f'shape output: {output.shape}') # (1, 896, 5313)
    print(f'dtype output: {output.dtype}') 
    print(f"Device seq is stored on: {output.device}\n")

    print('embedding information')
    print(f'type embeddings: {type(embeddings)}')
    print(f"Shape embeddings: {embeddings.shape}")
    print(f"Datatype embeddings: {embeddings.dtype}") #float32
    print(f"Device embeddings is stored on: {embeddings.device}\n")

    embeddings_tensor[i] = embeddings
    output_tensor[i] = output

    torch.save(embeddings, f'/exports/humgen/idenhond/enformer_dev/enformer-pytorch/data/data_tensors_test/embeddings_seq{i}.pt')
    torch.save(output, f'/exports/humgen/idenhond/enformer_dev/enformer-pytorch/data/data_tensors_test/output_seq{i}.pt')

print(f"Shape of big tensor embeddings: {embeddings_tensor.shape}")
print(f"Shape of big tensor output: {output_tensor.shape}")
print(f"dtype of big tensor embeddings: {embeddings_tensor.dtype}")
print(f"dtype of big tensor output: {output_tensor.dtype}")
print(f"Device big tensor embeddings is stored on: {embeddings_tensor.device}")
print(f"Device big tensor output is stored on: {output_tensor.device}\n")


torch.save(embeddings_tensor, f'/exports/humgen/idenhond/enformer_dev/enformer-pytorch/data/data_tensors_test/embeddings_big.pt')
torch.save(output_tensor, f'/exports/humgen/idenhond/enformer_dev/enformer-pytorch/data/data_tensors_test/output_big.pt')


print(f'Time: {datetime.now() - start}') 













"""
# OPTIE 1

# initialize empty torch to store all sequences in
seq = torch.empty(size=((length), 196608))


# add all sequences to the torch
for i in range((length)):
    print(f'i option 1: {i}')
    seq[i] = ds[i]
    

# sequence information
print(f"Datatype of tensor seq: {seq.dtype}") #float32 denk ik
print(f'type seq: {type(seq)}') 
seq = seq.type(torch.int64)
print(f"Shape of tensor seq: {seq.shape}")
print(f"Datatype of tensor seq: {seq.dtype}") #int64
print(f"Device tensor seq is stored on: {seq.device}")  #cpu
print('\n')

output_opt1, embeddings_opt1 = model(seq, return_embeddings = True, head = 'human')
print(f"Shape of tensor output option 1: {output_opt1.shape}")
print(f"Shape of tensor embeddings option 1: {embeddings_opt1.shape}")
print(f"dtype of tensor output option 1: {output_opt1.dtype}")
print(f"dtype of tensor embeddings option 1: {embeddings_opt1.dtype}")

# eind optie 1
"""