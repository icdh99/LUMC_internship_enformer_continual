from enformer_pytorch import Enformer, GenomeIntervalDataset
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import kipoiseq
from kipoiseq import Interval
import joblib
import gzip
import pyfaidx
import pandas as pd
import matplotlib as mpl

### Predictions with enformer-pytorch model
df_targets = pd.read_csv('/exports/humgen/idenhond/data/Basenji/human-targets.txt', sep = '\t')
print(df_targets.head(10))

print(f'predicting with Enformer pytorch model')

enformer = Enformer.from_pretrained('EleutherAI/enformer-official-rough')
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

def plot_tracks(tracks, interval, name, height=1.5):
  fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True)
  for ax, (title, y) in zip(axes, tracks.items()):
    # ax.fill_between(np.linspace(interval.start, interval.end, num=len(y)), y)
    ax.fill_between(np.linspace(35082742, 35197430, num=len(y)), y, color = 'red')
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

seq = ds[0] # (196608,)
print(seq.shape)

pred = enformer(seq, head = 'human').detach().numpy()
print(pred.shape)

tracks = {'DNASE:CD14-positive monocyte female': pred[:, 41],
          'DNASE:keratinocyte female': pred[:, 42],
          'CHIP:H3K27ac:keratinocyte female': pred[:, 706],
          'CAGE:Keratinocyte - epidermal': np.log10(1 + pred[:, 4799])}
figname = 'test_enformer_pytorch_red'

plot_tracks(tracks, None, figname)

