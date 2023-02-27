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
subset = str(sys.argv[1])
print(f'generating enformer output for {subset} sequences')

colnames=['chr', 'start', 'end', 'category'] 
df = pd.read_csv('/exports/humgen/idenhond/data/Basenji/sequences.bed', sep='\t', names=colnames)
print(f'number of sequences: {df.shape}')
df_subset = df[df['category'] == subset].reset_index(drop=True)
print(f'number of {subset} sequences: {df_subset.shape}')

# if not os.path.exists(f'/exports/humgen/idenhond/projects/enformer/get_output_emb/tmp_bed_{subset}'):
#     os.mkdir(f'tmp_bed_{subset}')
#     print(f'directory tmp_bed_{subset} is created')
# else: 
#     print(f'directory tmp_bed_{subset} already exists')


"""
load necessary functions and objects only once
"""

filter_train = lambda df: df.filter(pl.col('column_4') == subset)   

# load model
# model = Enformer.from_hparams(
#     dim = 1536,
#     depth = 11,
#     heads = 8,
#     output_heads = dict(human = 5313, mouse = 1643),
#     target_length = 896 ).cuda()

## TODO: change this to pretrained model from huggingface
model = Enformer.from_pretrained("EleutherAI/enformer-official-rough")
model = model.eval().cuda()



"""
try to get output and embeddings for all sequences at the same time using torch.no_grad
"""
bed_file = '/exports/humgen/idenhond/data/Basenji/sequences.bed'
ds = GenomeIntervalDataset(
    bed_file = bed_file,
    fasta_file = '/exports/humgen/idenhond/genomes/hg38.ml.fa',                  
    filter_df_fn = filter_train,                        
    return_seq_indices = True,                          
    shift_augs = None,                              
    context_length = 196_608,
    chr_bed_to_fasta_map = {} )

print(f'number of sequences in ds object: {len(ds)}')


length = len(ds)

seq = torch.empty(size=(length, 196608))
for i in range(length):
    seq[i] = ds[i].cuda()

print(seq.shape)
print(seq[0].shape)

with torch.no_grad():
  output, embeddings = model(seq[1], return_embeddings = True, head = 'human')
#   print(output)
#   print(embeddings)
  print(output.shape)
  print(embeddings.shape)

exit()

"""
from store_tensors_pretrainedmodel.py:
store output and embeddings for each test/valid sequence seperately
"""

t = 0 
for row in df_subset.itertuples():
    t += 1
    print(t, row)

    with open(f'tmp_bed_{subset}/tmp_{subset}.bed', 'w') as f:
        f.truncate(0)
        line = [row.chr, str(row.start), str(row.end), str(subset), '\n']
        f.write('\t'.join(line))

    # load sequences
    ds = GenomeIntervalDataset(
        bed_file = f'tmp_bed_{subset}/tmp_{subset}.bed',
        fasta_file = '/exports/humgen/idenhond/genomes/hg38.ml.fa',                  
        filter_df_fn = filter_train,                        
        return_seq_indices = True,                          
        shift_augs = None,                              
        context_length = 196_608,
        chr_bed_to_fasta_map = {} )
    # print(f'number of sequences in ds object: {len(ds)}')

    seq = ds[0].cuda()
    # print(type(seq))
    # print(seq.device)
    # print(seq.shape)
    
    # print('seq information')
    # print(f'type seq: {type(seq)}')
    # print(f"Shape seq: {seq.shape}")
    # print(f"Datatype seq: {seq.dtype}") #int64
    # print(f"Device seq is stored on: {seq.device}\n")

    # output, embeddings = model(seq, return_embeddings = True, head = 'human')
    with torch.no_grad():
        output = model(seq)['human']

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

    # if subset == 'valid':
        # torch.save(embeddings, f'/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_embeddings_newmodel/embeddings_seq{t}.pt')
        # torch.save(output, f'/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_embeddings_newmodel/output_seq{t}.pt')

    # if subset == 'test':
        # torch.save(embeddings, f'/exports/humgen/idenhond/data/Enformer_test/Enformer_test_embeddings_newmodel/embeddings_seq{t}.pt')
        # torch.save(output, f'/exports/humgen/idenhond/data/Enformer_test/Enformer_test_embeddings_newmodel/output_seq{t}.pt')


print(f'Time: {datetime.now() - start}') 
