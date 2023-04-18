from datetime import datetime
start = datetime.now()
import pandas as pd
import pytorch_lightning as pytorchlightning
import numpy as np
import torch.nn as nn
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

if not os.path.exists(f'/exports/humgen/idenhond/projects/enformer/get_output_emb/tmp_bed_{subset}'):
    os.mkdir(f'tmp_bed_{subset}')
    print(f'directory tmp_bed_{subset} is created')
else: 
    print(f'directory tmp_bed_{subset} already exists')

"""
load necessary functions and objects only once
"""

filter_train = lambda df: df.filter(pl.col('column_4') == subset)   

class model(pytorchlightning.LightningModule):
	def __init__(self):
		# define model
		super(model, self).__init__()
		self.linear = nn.Linear(in_features = 3072, out_features = 5313, bias = True)
		self.softplus = nn.Softplus(beta = 1, threshold = 20)	# default values for nn.Softplus()
		self.lr = 1e-4
		self.loss = nn.PoissonNLLLoss()
		self.train_log = []
		self.save_hyperparameters()

	def forward(self, x):
		# define forward pass
		x = self.linear(x)
		x = self.softplus(x)
		return x

	def configure_optimizers(self):
		# define optimizer 
		return torch.optim.Adam(self.parameters(), lr = self.lr)

	def training_step(self, train_batch, batch_idx):
		# define training loop steps
		x, y = train_batch 
		logits = self.forward(x)
		loss = self.loss(logits, y)
		self.log("train_loss", loss, on_epoch=True, prog_bar=True)
		self.train_log.append(loss.cpu().detach().numpy())
		# tb_logger = self.logger.experiment
		# tb_logger.add_scalars("losses", {"train_loss": loss})
		self.logger.experiment.add_scalars('loss', {'train': loss},self.global_step) 
		return loss
	
	def test_step(self, batch, batch_idx):
		x, y = batch
		logits = self.forward(x)
		test_loss = self.loss(logits, y)
		self.log("test_loss", test_loss, on_epoch=True, prog_bar=True)
		return test_loss

	def validation_step(self, valid_batch, batch_idx):
		# define validation loop steps
		x, y = valid_batch
		logits = self.forward(x)
		val_loss = self.loss(logits, y)
		self.log("val_loss", val_loss, prog_bar=True)
		self.logger.experiment.add_scalars('loss', {'valid': val_loss},self.global_step)
		# self.logger.experiment.add_scalars("losses", {"val_loss": val_loss})

		return val_loss

	def predict_step(self, batch, batch_idx):
		x, y = batch
		return self(x), y
# load model
path = '/exports/humgen/idenhond/projects/enformer/dnn_head/dnn_head_train/model_2023-03-10 17:52:03.039827_v3/epoch=19-step=5320-val_loss=0.8.ckpt'
model = model.load_from_checkpoint(path)
model.eval().cuda()
# model = model.eval()

"""
store output and embeddings for each test/valid sequence seperately
"""

t = 0 
for row in df_subset.itertuples():
    t += 1
    # print(t, row)

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
    # seq = ds[0]
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
        output = model(seq)
        print(f'output shape: {output.shape}')
    if t == 1:
        break


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

    if subset == 'valid':
        # torch.save(embeddings, f'/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_embeddings_newmodel/embeddings_seq{t}.pt')
        torch.save(output, f'/exports/humgen/idenhond/data/Enformer_output_dnn_head_v3/validation_output/output_seq{t}.pt')

    #     # torch.save(output, f'/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_embeddings_newmodel/output_seq{t}.pt')

    if subset == 'test':
    #     torch.save(embeddings, f'/exports/humgen/idenhond/data/Enformer_test/Enformer_test_embeddings_newmodel/embeddings_seq{t}.pt')
        torch.save(output, f'/exports/humgen/idenhond/data/Enformer_output_dnn_head_v3/test_output/output_seq{t}.pt')

    if subset == 'train':
        torch.save(output, f'/exports/humgen/idenhond/data/Enformer_output_dnn_head_v3/train_output/output_seq{t}.pt')

print(f'Time: {datetime.now() - start}') 
