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
print(f'torch.cuda_is_available(): {torch.cuda.is_available()}\n')
#Additional Info when using cuda
if device.type == 'cuda':
    print(f'Device name: {torch.cuda.get_device_name(0)}')
    print('Memory Usage:', 'Allocated', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB,', 'Cached', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB\n')
    # print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    # print('Cached:', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')   # torch.cuda.memory_cached has been renamed to torch.cuda.memory_reserved

"""
Test Enformer model
input sequence derived from fasta file
"""
from enformer_pytorch import Enformer, seq_indices_to_one_hot, GenomeIntervalDataset
import polars as pl

filter_train = lambda df: df.filter(pl.col('column_4') == 'valid')        
# filter_train = lambda df: df    # heb ik niet nodig met mijn bed file op dit moment

ds = GenomeIntervalDataset(
    bed_file = '/exports/humgen/idenhond/Enformer_data_google/data_human_sequences.bed',
    # bed_file = '/exports/humgen/idenhond/enformer_dev/enformer-pytorch/data_test/test_enformer_fastainput.bed',    
    # bed_file = '/exports/humgen/idenhond/enformer_dev/enformer-pytorch/data_test/test_enformer_100seqs.bed',                   # bed file - columns 0, 1, 2 must be <chromosome>, <start position>, <end position>
    # fasta_file = '/exports/genomes/species/H.sapiens/GRCh38/reference.fa',   # file seems empty 
    fasta_file = '/exports/humgen/idenhond/genomes/hg38.ml.fa',                    # path to fasta file
    filter_df_fn = filter_train,                        # filter dataframe function --> nu niet nodig
    return_seq_indices = True,                          # return nucleotide indices (ACGTN) or one hot encodings
    shift_augs = (-2, 2),                               # random shift augmentations from -2 to +2 basepairs --> even checken wat dit doet
    context_length = 196_608,
    # this can be longer than the interval designated in the .bed file,
    # in which case it will take care of lengthening the interval on either sides
    # as well as proper padding if at the end of the chromosomes
    chr_bed_to_fasta_map = {    # not needed --> empty
        # 'chr1': 'chromosome1',  # if the chromosome name in the .bed file is different than the key name in the fasta file, you can rename them on the fly
        # 'chr2': 'chromosome2',
        # 'chr3': 'chromosome3',
        # etc etc
    }
)

print(f'number of sequences in ds object: {len(ds)}')


# load model
model = Enformer.from_hparams(
    dim = 1536,
    depth = 11,
    heads = 8,
    output_heads = dict(human = 5313, mouse = 1643),
    target_length = 896,
)

# number of sequences to generate model output for
length = 4
# length = len(ds)

# OPTIE 1

# # initialize empty torch to store all sequences in
# seq = torch.empty(size=((length), 196608))

# # add all sequences to the torch
# for i in range((length)):
#     seq[i] = ds[i]

# # sequence information
# print(f'type seq: {type(seq)}')
# seq = seq.type(torch.int64)
# print(f'type seq: {type(seq)}')
# print(f"Shape of tensor seq: {seq.shape}")
# print(f"Datatype of tensor seq: {seq.dtype}")
# print(f"Device tensor seq is stored on: {seq.device}")
# print('\n')

# output, embeddings = model(seq, return_embeddings = True, head = 'human')
# print(f"Shape of tensor output option 1: {output.shape}")
# print(f"Shape of tensor embeddings option 1: {embeddings.shape}")

# eind optie 1


# OPTIE 2 
embeddings_tensor_opt2 = torch.empty(size = (length, 896, 3072))
output_tensor_opt2 = torch.empty(size = (length, 896, 5313))

seq = None 
for i in range(length):
    seq = ds[i]
    print(i)

    output, embeddings = model(seq, return_embeddings = True, head = 'human') # --> wordt wel gebruikt in tutorial. maar dan moet je verderop de human/mouse dingen aanpassen
    print('seq information')
    print(f'type seq: {type(seq)}')
    print(f"Shape of tensor seq: {seq.shape}")
    print(f"Datatype of tensor seq: {seq.dtype}")
    print(f"Device tensor seq is stored on: {seq.device}")

    print(f'output information')
    print(f'type output: {type(output)}')
    print(output.shape) # (1, 896, 5313)
    # print((output['mouse']).shape) # (1, 896, 1643)

    print('embedding information')
    print(f'type embeddings: {type(embeddings)}')
    print(f"Shape of tensor embeddings: {embeddings.shape}")
    print(f"Datatype of tensor embeddings: {embeddings.dtype}")
    print(f"Device tensor embeddings is stored on: {embeddings.device}")

    embeddings_tensor_opt2[i] = embeddings
    output_tensor_opt2[i] = output


    print('\n')

print(f"Shape of big tensor embeddings option 2: {embeddings_tensor_opt2.shape}")
print(f"Shape of big tensor output option 2: {output_tensor_opt2.shape}")

print(output_tensor_opt2 == output)
print(embeddings_tensor_opt2 == embeddings)

# # print(output.shape) # (1, 896, 5313)
# # print(f"Shape of tensor embeddings: {embeddings.shape}")

print(f'Time: {datetime.now() - start}') 