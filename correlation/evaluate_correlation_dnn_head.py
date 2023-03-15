from datetime import datetime
start = datetime.now()
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import tensorflow as tf
import os 
import json
import pandas as pd
import pyfaidx
import kipoiseq
import functools
import sys
from kipoiseq import Interval
import torch
from enformer_pytorch import Enformer
from enformer_pytorch.metrics import MeanPearsonCorrCoefPerChannel
from tqdm import tqdm
from torchmetrics.regression.pearson import PearsonCorrCoef

subset = str(sys.argv[1])
max_steps = int(sys.argv[2])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device (set to GPU if available):', device)

# load stored outputs
if subset == 'test':
  print('you have said test')
  # tfr_file = f'/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output_newmodel/output_test.pt'
  tfr_file = f'/exports/archive/hg-funcgenom-research/idenhond/Enformer_test/Enformer_test_embeddings_newmodel/embeddings_test_pretrainedmodel.pt'
if subset == 'valid':
  print('you have said valid')
  tfr_file = f'/exports/archive/hg-funcgenom-research/idenhond/Enformer_validation/Enformer_validation_embeddings_newmodel/embeddings_validation_pretrainedmodel.pt'
  # tfr_file = f'/exports/humgen/idenhond/data/Enformer_validation/output_validation.pt'

  
# tensor_out = torch.load(tfr_file, map_location=torch.device(device)) 
# print(f'device of output tensor: {tensor_out.device}')
# print(f'shape of output tensor: {tensor_out.shape}\n')

print(f'Time after loading output tensor: {datetime.now() - start}') 


SEQUENCE_LENGTH = 196_608
BIN_SIZE = 128
TARGET_LENGTH = 896

"""
base dir is now folder data/Basenji with human stuff
TODO: make into seperate directories for human and mouse, similar to google cloud storage
"""

human_fasta_path = '/exports/humgen/idenhond/genomes/hg38.ml.fa'
mouse_fasta_path = '/exports/humgen/idenhond/genomes/mm10.ml.fa'
if not os.path.isfile(human_fasta_path): print('please supply the human genome fasta file')
if not os.path.isfile(mouse_fasta_path): print('please supply the mouse genome fasta file')

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
  
class FastaStringExtractor:
    # this class works with human_fasta_path and mouse_fasta_path as fasta_file variable
    def __init__(self, fasta_file):
        self.fasta = pyfaidx.Fasta(fasta_file)
        self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}

    def extract(self, interval: Interval, **kwargs) -> str:
        # Truncate interval if it extends beyond the chromosome lengths.
        chromosome_length = self._chromosome_sizes[interval.chrom]
        trimmed_interval = Interval(interval.chrom,
                                    max(interval.start, 0),
                                    min(interval.end, chromosome_length),
                                    )
        # pyfaidx wants a 1-based interval
        sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
                                          trimmed_interval.start + 1,
                                          trimmed_interval.stop).seq).upper()
        # Fill truncated values with N's.
        pad_upstream = 'N' * max(-interval.start, 0)
        pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
        return pad_upstream + sequence + pad_downstream

    def close(self):
        return self.fasta.close()

class BasenjiDataSet(torch.utils.data.IterableDataset):
  @staticmethod
  def get_organism_path(organism):
    # return os.path.join('gs://basenji_barnyard/data', organism) # change to human
    return '/exports/humgen/idenhond/data/Basenji'
  @classmethod
  def get_metadata(cls, organism):
    # Keys:
    # num_targets, train_seqs, valid_seqs, test_seqs, seq_length,
    # pool_width, crop_bp, target_length
    path = os.path.join(cls.get_organism_path(organism), 'statistics.json')
    path = '/exports/humgen/idenhond/data/Basenji/statistics.json'
    with tf.io.gfile.GFile(path, 'r') as f:
      return json.load(f)
  @staticmethod
  def one_hot_encode(sequence):
    return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

  @classmethod
  def get_tfrecord_files(cls, organism, subset):
    # Sort the values by int(*).
    path_to_tfr_records = os.path.join(
        cls.get_organism_path(organism), 'tfrecords', f'{subset}-*.tfr'
      )
    if subset == 'train':
        path_to_tfr_records = f'/exports/archive/hg-funcgenom-research/idenhond/Basenji/tfrecords/{subset}-*.tfr'
    else:
        path_to_tfr_records = f'/exports/humgen/idenhond/data/Basenji/tfrecords/{subset}-*.tfr'
    print(f"output of get_tfrecord_files function: \n{sorted(tf.io.gfile.glob(path_to_tfr_records), key=lambda x: int(x.split('-')[-1].split('.')[0]))}")
    return sorted(tf.io.gfile.glob(path_to_tfr_records), key=lambda x: int(x.split('-')[-1].split('.')[0]))
  
  @property
  def num_channels(self):
    metadata = self.get_metadata(self.organism)
    # num_targets = 5313 for human
    return metadata['num_targets']

  @staticmethod
  def deserialize(serialized_example, metadata):
    """Deserialize bytes stored in TFRecordFile."""
    # Deserialization
    feature_map = {
        'sequence': tf.io.FixedLenFeature([], tf.string),  # Ignore this, resize our own bigger one
        'target': tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_example(serialized_example, feature_map)
    sequence = tf.io.decode_raw(example['sequence'], tf.bool)
    sequence = tf.reshape(sequence, (metadata['seq_length'], 4))
    sequence = tf.cast(sequence, tf.float32)

    target = tf.io.decode_raw(example['target'], tf.float16)
    target = tf.reshape(target,
                        (metadata['target_length'], metadata['num_targets']))
    target = tf.cast(target, tf.float32)

    return {'sequence_old': sequence,
            'target': target}

  @classmethod
  def get_dataset(cls, organism, subset, num_threads=8):
    metadata = cls.get_metadata(organism)
    print(f'metadata in get_dataset_function: \n {metadata}')
    dataset = tf.data.TFRecordDataset(cls.get_tfrecord_files(organism, subset),
                                      compression_type='ZLIB',
                                      num_parallel_reads=num_threads).map(
                                          functools.partial(cls.deserialize, metadata=metadata)
                                      )
    return dataset

  def __init__(self, organism:str, subset:str, seq_len:int, fasta_path:str, n_to_test:int = -1):
    assert subset in {"train", "valid", "test"}
    assert organism in {"human", "mouse"}
    self.organism = organism
    self.subset = subset
    self.base_dir = self.get_organism_path(organism)
    self.seq_len = seq_len
    self.fasta_reader = FastaStringExtractor(fasta_path)
    self.n_to_test = n_to_test
    with tf.io.gfile.GFile(f"{self.base_dir}/sequences.bed", 'r') as f:
      region_df = pd.read_csv(f, sep="\t", header=None)
      region_df.columns = ['chrom', 'start', 'end', 'subset']
      self.region_df = region_df.query('subset==@subset').reset_index(drop=True)
      
  def __iter__(self):
    worker_info = torch.utils.data.get_worker_info()
    assert worker_info is None, "Only support single process loading"
    # If num_threads > 1, the following will actually shuffle the inputs! luckily we catch this with the sequence comparison
    basenji_iterator = self.get_dataset(self.organism, self.subset, num_threads=1).as_numpy_iterator()
    for i, records in enumerate(basenji_iterator):
      loc_row = self.region_df.iloc[i]
      target_interval = Interval(loc_row['chrom'], loc_row['start'], loc_row['end'])
      sequence_one_hot = self.one_hot_encode(self.fasta_reader.extract(target_interval.resize(self.seq_len)))
      if self.n_to_test >= 0 and i < self.n_to_test:
        old_sequence_onehot = records["sequence_old"]
        if old_sequence_onehot.shape[0] > sequence_one_hot.shape[0]:
          diff = old_sequence_onehot.shape[0] - sequence_one_hot.shape[0]
          trim = diff//2
          np.testing.assert_equal(old_sequence_onehot[trim:(-trim)], sequence_one_hot)
        elif sequence_one_hot.shape[0] > old_sequence_onehot.shape[0]:
          diff = sequence_one_hot.shape[0] - old_sequence_onehot.shape[0]
          trim = diff//2
          np.testing.assert_equal(old_sequence_onehot, sequence_one_hot[trim:(-trim)])
        else:
          np.testing.assert_equal(old_sequence_onehot, sequence_one_hot)
      yield {
          "sequence": sequence_one_hot,
          "target": records["target"],
      }

# model = Enformer.from_pretrained("EleutherAI/enformer-official-rough")
# model = model.eval().cuda()

# load model
path = '/exports/humgen/idenhond/projects/enformer/dnn_head/dnn_head_train/model_2023-03-10 17:52:03.039827_v3/epoch=19-step=5320-val_loss=0.8.ckpt'
model = model.load_from_checkpoint(path)
model.eval().cuda()

print(f'Time after loading model: {datetime.now() - start}') 

# def compute_correlation(model, organism:str="human", subset:str="valid", max_steps=-1):
def compute_correlation(model, organism:str="human", subset:str=subset, max_steps=max_steps):
  print(f'organism: {organism}')
  print(f'subset: {subset}')
  fasta_path = human_fasta_path if organism == "human" else mouse_fasta_path
  print(f'fasta path: {fasta_path}')
  ds = BasenjiDataSet(organism, subset, SEQUENCE_LENGTH, fasta_path)
  total = len(ds.region_df) # number of records
  print(f'number of records: {total}')
  dl = torch.utils.data.DataLoader(ds, num_workers=0, batch_size=1)
  corr_coef = MeanPearsonCorrCoefPerChannel(n_channels=ds.num_channels) # 5313 channels
  n_steps = total if max_steps <= 0 else max_steps
  print(f'number of steps to calculate correlation coefficient: {n_steps}')
  for i,batch in enumerate(tqdm(dl, total=n_steps)):
    if max_steps > 0 and i >= max_steps:
      break
    batch_gpu = {k:v.to(model.device) for k,v in batch.items()}
    sequence = batch_gpu['sequence']
    # print(f'sequence type: {type(sequence)}')
    # print(f'sequence shape: {sequence.shape}') # torch.Size([1, 196608, 4])
    # print(f'sequence device: {sequence.device}')
    target = batch_gpu['target']
    # print(f'target type: {type(target)}')
    # print(f'target shape: {target.shape}')  # torch.Size([1, 896, 5313])
    # print(f'target device: {target.device}')
    with torch.no_grad():
      tensor_out = torch.load(f'/exports/humgen/idenhond/data/Enformer_train/Enformer_train_embeddings_pretrainedmodel/embeddings_seq{i+1}.pt', map_location=torch.device(device))
      seq_emb = torch.unsqueeze(tensor_out, 0)
      # pred = model(seq_emb)[organism]
      pred = model(seq_emb)
      # print(f'pred type: {type(pred)}')
      # print(f'pred shape: {pred.shape}')  # torch.Size([1, 896, 5313])
      # print(f'pred device: {pred.device}')

      # print(f'is predicted tensor equal to output tensor: {torch.equal(tensor_out_one, pred.cpu())}')
      corr_coef(preds=pred.cpu(), target=target.cpu())

      # compu = corr_coef.compute()
      # print(f'{i} corr coef compute: {compu}')
      # print(f'{i} shape corr coef compute: {compu.shape} ')

  
  compu = corr_coef.compute()
  print(f'final corr coef compute: {compu}')
  print(f'final shape corr coef compute: {compu.shape} ')
  # print(f'the mean correlation coefficient for {organism} {subset} sequences calculated over {n_steps} sequence-target sets is {corr_coef.compute().mean()}')
  print(f'shape mean correlation coefficient: {corr_coef.compute().mean().shape}')

  # DIT IS NU HELEMAAL OVERSCHREVEN
  t_np = compu.numpy()
  print(t_np.shape)
  df = pd.DataFrame(t_np)
  df.to_csv(f"/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_{subset}_dnn_head.csv",index=True)
  return corr_coef.compute().mean()

a = compute_correlation(model, organism="human", subset=subset, max_steps=max_steps)
print(f'mean correlation for {subset}: {a}')

print(f'Time: {datetime.now() - start}') 