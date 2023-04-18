import pandas as pd
import tensorflow as tf
"""
Notebook enformer-pytorch output
"""
subset = 'valid'
filepath = '/exports/humgen/idenhond/data/Basenji/sequences.bed'
with tf.io.gfile.GFile(filepath, 'r') as f:
      region_df = pd.read_csv(f, sep="\t", header=None)
      region_df.columns = ['chrom', 'start', 'end', 'subset']
      region_df = region_df.query('subset==@subset').reset_index(drop=True)

print(region_df.shape)

"""
own output
"""
colnames=['chr', 'start', 'end', 'category'] 
colnames = ['chrom', 'start', 'end', 'subset']

df = pd.read_csv('/exports/humgen/idenhond/data/Basenji/sequences.bed', sep='\t', names=colnames)
df_test = df[df['subset'] == 'valid'].reset_index(drop=True)

print(df_test.shape)

print(region_df)
print(df_test)

print(region_df.compare(df_test, keep_equal = True))
print(region_df == df_test)

