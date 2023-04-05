import matplotlib.pyplot as plt
import seaborn as sns
import kipoiseq
from kipoiseq import Interval
import tensorflow_hub as hub
import joblib
import gzip
import kipoiseq
from kipoiseq import Interval
import pyfaidx
import pandas as pd
import numpy as np
import matplotlib as mpl
import tensorflow as tf

transform_path = 'gs://dm-enformer/models/enformer.finetuned.SAD.robustscaler-PCA500-robustscaler.transform.pkl'
model_path = 'https://tfhub.dev/deepmind/enformer/1'
# fasta_file = '/root/data/genome.fa'
fasta_file = '/exports/humgen/idenhond/genomes/hg38.ml.fa'
clinvar_vcf = '/root/data/clinvar.vcf.gz'

df_targets = pd.read_csv('/exports/humgen/idenhond/data/Basenji/human-targets.txt', sep = '\t')
print(df_targets.head(10))

target_interval = kipoiseq.Interval('chr11', 35_082_742, 35_197_430)  # @param


# sequence

# embedding

# dnn head model 

# output


# code from notebook

SEQUENCE_LENGTH = 393216

class Enformer:
  def __init__(self, tfhub_url):
    self._model = hub.load(tfhub_url).model

  def predict_on_batch(self, inputs):
    predictions = self._model.predict_on_batch(inputs)
    return {k: v.numpy() for k, v in predictions.items()}

  @tf.function
  def contribution_input_grad(self, input_sequence,
                              target_mask, output_head='human'):
    input_sequence = input_sequence[tf.newaxis]

    target_mask_mass = tf.reduce_sum(target_mask)
    with tf.GradientTape() as tape:
      tape.watch(input_sequence)
      prediction = tf.reduce_sum(
          target_mask[tf.newaxis] *
          self._model.predict_on_batch(input_sequence)[output_head]) / target_mask_mass

    input_grad = tape.gradient(prediction, input_sequence) * input_sequence
    input_grad = tf.squeeze(input_grad, axis=0)
    return tf.reduce_sum(input_grad, axis=-1)
  
class FastaStringExtractor:
    
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

def one_hot_encode(sequence):
  return kipoiseq.transforms.functional.one_hot_dna(sequence).astype(np.float32)

def plot_tracks(tracks, interval, name, height=1.5):
  fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
  for ax, (title, y) in zip(axes, tracks.items()):
    ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y)
    ax.set_title(title)
    sns.despine(top=True, right=True, bottom=True)
  ax.set_xlabel(str(interval))
  plt.gcf().get_axes()[0].set_ylim(0, 3)
  plt.gcf().get_axes()[1].set_ylim(0, 3)
  plt.gcf().get_axes()[2].set_ylim(0, 70)
  plt.gcf().get_axes()[3].set_ylim(0, 3)
  plt.tight_layout()
  plt.savefig(f'{name}.png')


### Predictions with Enformer model
print(f'predicting with Enformer model')
model = Enformer(model_path)
fasta_extractor = FastaStringExtractor(fasta_file)
target_interval = kipoiseq.Interval('chr11', 35_082_742, 35_197_430)  # @param
sequence_one_hot = one_hot_encode(fasta_extractor.extract(target_interval.resize(SEQUENCE_LENGTH)))
print(sequence_one_hot.shape)
predictions = model.predict_on_batch(sequence_one_hot[np.newaxis])['human'][0]
print(predictions.shape)
print(type(predictions))

tracks = {'DNASE:CD14-positive monocyte female': predictions[:, 41],
          'DNASE:keratinocyte female': predictions[:, 42],
          'CHIP:H3K27ac:keratinocyte female': predictions[:, 706],
          'CAGE:Keratinocyte - epidermal': np.log10(1 + predictions[:, 4799])}
figname = 'test_enformer_usage'
plot_tracks(tracks, target_interval, figname)


