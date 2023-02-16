from datetime import datetime
start = datetime.now()
import pandas as pd
import sys
import os
print(f'{start} Start of Python script {sys.argv[0]}')
import torch
from enformer_pytorch import Enformer, seq_indices_to_one_hot, GenomeIntervalDataset
import polars as pl

"""
Information about CPU & GPU usage from PyTorch
"""
# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device (set to GPU if available):', device)

"""
Prepare dataframe 
"""
colnames=['chr', 'start', 'end', 'category'] 
df = pd.read_csv('/exports/humgen/idenhond/Enformer_data_google/data_human_sequences.bed', sep='\t', names=colnames)
print(f'number of sequences: {df.shape}')
df_test = df[df['category'] == 'test'].reset_index(drop=True)
print(f'number of test sequences: {df_test.shape}')

if not os.path.exists('/exports/humgen/idenhond/enformer_dev/enformer-pytorch/scripts/tmp_bed_test'):
    os.mkdir('tmp_bed_test')
    print('directory tmp_bed_test is created')
else: 
    print('directory tmp_bed_test already exists')


"""
load necessary functions and objects only once
"""

filter_train = lambda df: df.filter(pl.col('column_4') == 'test')   

# load model
model = Enformer.from_hparams(
    dim = 1536,
    depth = 11,
    heads = 8,
    output_heads = dict(human = 5313, mouse = 1643),
    target_length = 896 )

"""
store output and embeddings for each test sequence seperately
"""

t = 0 
for row in df_test.itertuples():
    t += 1
    print(t, row)

    with open('tmp_bed_test/tmp_test.bed', 'w') as f:
        f.truncate(0)
        line = [row.chr, str(row.start), str(row.end), 'test', '\n']
        f.write('\t'.join(line))

    # load sequences
    ds = GenomeIntervalDataset(
        bed_file = 'tmp_bed_test/tmp_test.bed',
        fasta_file = '/exports/humgen/idenhond/genomes/hg38.ml.fa',                  
        filter_df_fn = filter_train,                        
        return_seq_indices = True,                          
        shift_augs = None,                              
        context_length = 196_608,
        chr_bed_to_fasta_map = {} )
    # print(f'number of sequences in ds object: {len(ds)}')

    seq = ds[0]
    
    # print('seq information')
    # print(f'type seq: {type(seq)}')
    # print(f"Shape seq: {seq.shape}")
    # print(f"Datatype seq: {seq.dtype}") #int64
    # print(f"Device seq is stored on: {seq.device}\n")

    output, embeddings = model(seq, return_embeddings = True, head = 'human')

    # print(f'output information')
    # print(f'type output: {type(output)}')
    # print(f'shape output: {output.shape}') # (1, 896, 5313)
    # print(f'dtype output: {output.dtype}') 
    # print(f"Device seq is stored on: {output.device}\n")

    # print('embedding information')
    # print(f'type embeddings: {type(embeddings)}')
    # print(f"Shape embeddings: {embeddings.shape}")
    # print(f"Datatype embeddings: {embeddings.dtype}") #float32
    # print(f"Device embeddings is stored on: {embeddings.device}\n")

    torch.save(embeddings, f'/exports/humgen/idenhond/enformer_dev/enformer-pytorch/data/tensors_test/embeddings_seq{t}.pt')
    torch.save(output, f'/exports/humgen/idenhond/enformer_dev/enformer-pytorch/data/tensors_test/output_seq{t}.pt')



print(f'Time: {datetime.now() - start}') 
