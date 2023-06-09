from enformer_pytorch import Enformer, GenomeIntervalDataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import kipoiseq
from kipoiseq import Interval
import pytorch_lightning as pl
import torch
import torch.nn as nn
import joblib
import gzip
import pyfaidx
import pandas as pd
import matplotlib as mpl

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device (set to GPU if available):', device)
print(f'torch.cuda_is_available(): {torch.cuda.is_available()}\n')

### Predictions with enformer-pytorch model
df_targets = pd.read_csv('/exports/humgen/idenhond/data/Basenji/human-targets.txt', sep = '\t')
print(df_targets.head(10))

print(f'predicting with Enformer pytorch model')

enformer = Enformer.from_pretrained('EleutherAI/enformer-official-rough') # ADD OWN MODEL
filter_train = lambda df: df

with open('seq.bed', 'w') as f:
  line = ['chr11', '35082742', '35197430']
  f.write('\t'.join(line))

ds = GenomeIntervalDataset(
    bed_file = './seq.bed',                       # bed file - columns 0, 1, 2 must be <chromosome>, <start position>, <end position>
    fasta_file = '/exports/humgen/idenhond/genomes/hg38.ml.fa',                        # path to fasta file
    filter_df_fn = filter_train,                        # filter dataframe function
    return_seq_indices = True,                          # return nucleotide indices (ACGTN) or one hot encodings
    shift_augs = (-2, 2),                               # random shift augmentations from -2 to +2 basepairs
    context_length = 196_608,
    # this can be longer than the interval designated in the .bed file,
    # in which case it will take care of lengthening the interval on either sides
    # as well as proper padding if at the end of the chromosomes
    chr_bed_to_fasta_map = {
        'chr1': 'chromosome1',  # if the chromosome name in the .bed file is different than the key name in the fasta file, you can rename them on the fly
        'chr2': 'chromosome2',
        'chr3': 'chromosome3',
        # etc etc
    }
)

# GET EMBEDDINGS FROM THIS SEQUENCE WITH PRETRAINED MODEL
seq = ds[0] # (196608,)
print(seq.shape)
pred, emb = enformer(seq, return_embeddings = True, head = 'human')
print(emb.shape)

# GET PREDICTIONS WITH DNN HEAD MODEL
class model(pl.LightningModule):
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

path = '/exports/humgen/idenhond/projects/enformer/dnn_head/dnn_head_train/model_2023-03-23 09:36:14.367136/epoch=18-step=5054-val_loss=0.8.ckpt'  # retrain dnn head, finished on 27/3
print(path)
model = model.load_from_checkpoint(path)
model.eval().cuda()

emb = emb.cuda()
pred = model(emb)
print(pred.shape)
print(type(pred))
pred = pred.detach().cpu().numpy()


def plot_tracks(tracks, interval, name, height=1.5):
  fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
  for ax, (title, y) in zip(axes, tracks.items()):
    # ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y)
    ax.fill_between(np.linspace(35082742, 35197430, num=len(y)), y, color = 'black')
    # 35082742	35197430
    ax.set_title(title)
    sns.despine(top=True, right=True, bottom=True)
  # ax.set_xlabel(str(interval))
  ax.set_xlabel('chr11 35082742 35197430')
  # ax[0].set_ylim(top=3)
  # ax[1].set_ylim(top=3)
  # ax[2].set_ylim(top=200)
  # ax[3].set_ylim(top=3)
  plt.gcf().get_axes()[0].set_ylim(0, 3)
  plt.gcf().get_axes()[1].set_ylim(0, 3)
  plt.gcf().get_axes()[2].set_ylim(0, 70)
  plt.gcf().get_axes()[3].set_ylim(0, 3)
  plt.tight_layout()
  plt.savefig(f'{name}.png')

# seq = ds[0] # (196608,)
# print(seq.shape)

# pred = enformer(seq, head = 'human').detach().numpy()
# print(pred.shape)

tracks = {'DNASE:CD14-positive monocyte female': pred[:, 41],
          'DNASE:keratinocyte female': pred[:, 42],
          'CHIP:H3K27ac:keratinocyte female': pred[:, 706],
          'CAGE:Keratinocyte - epidermal': np.log10(1 + pred[:, 4799])}
figname = 'test_enformer_pytorchdnnhead_black'

plot_tracks(tracks, None, figname)

