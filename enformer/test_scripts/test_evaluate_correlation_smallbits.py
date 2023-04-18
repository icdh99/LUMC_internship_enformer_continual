import torch
import numpy as np
import tensorflow as tf
import os 
import json
import pandas as pd
import pyfaidx
import kipoiseq
import functools
from kipoiseq import Interval

# class FastaStringExtractor:
    
#     def __init__(self, fasta_file):
#         self.fasta = pyfaidx.Fasta(fasta_file)
#         self._chromosome_sizes = {k: len(v) for k, v in self.fasta.items()}
#         print({k: len(v) for k, v in self.fasta.items()})

#     def extract(self, interval: Interval, **kwargs) -> str:
#         # Truncate interval if it extends beyond the chromosome lengths.
#         chromosome_length = self._chromosome_sizes[interval.chrom]
#         trimmed_interval = Interval(interval.chrom,
#                                     max(interval.start, 0),
#                                     min(interval.end, chromosome_length),
#                                     )
#         # pyfaidx wants a 1-based interval
#         sequence = str(self.fasta.get_seq(trimmed_interval.chrom,
#                                           trimmed_interval.start + 1,
#                                           trimmed_interval.stop).seq).upper()
#         # Fill truncated values with N's.
#         pad_upstream = 'N' * max(-interval.start, 0)
#         pad_downstream = 'N' * max(interval.end - chromosome_length, 0)
#         return pad_upstream + sequence + pad_downstream

#     def close(self):
#         return self.fasta.close()
# # test class
# human_fasta_path = '/exports/humgen/idenhond/genomes/hg38.ml.fa'
# mouse_fasta_path = '/exports/humgen/idenhond/genomes/mm10.ml.fa'
# fasta_path = human_fasta_path
# fasta_path = mouse_fasta_path
# fasta_reader = FastaStringExtractor(fasta_path)

# path = '/exports/humgen/idenhond/data/Basenji/statistics.json'
# with tf.io.gfile.GFile(path, 'r') as f:
#     print(json.load(f))

# subset = 'valid'
# path_to_tfr_records = f'/exports/humgen/idenhond/data/Basenji/tfrecords/{subset}-*.tfr'
# print(sorted(tf.io.gfile.glob(path_to_tfr_records), key=lambda x: int(x.split('-')[-1].split('.')[0])))

