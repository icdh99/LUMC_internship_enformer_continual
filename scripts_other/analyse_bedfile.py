"""
Count number of train, test and validation sequences of Enformer data
Bed file supplied by google
"""
import pandas as pd    

colnames=['chr', 'start', 'end', 'category'] 
df = pd.read_csv('/exports/humgen/idenhond/Enformer_data_google/data_human_sequences.bed', sep='\t', names=colnames)

print(df.shape)
print(df.head())
print(df['category'].value_counts())

# df['length'] = df['end'] - df['start'] # each sequence has the same length
print(df.head())
