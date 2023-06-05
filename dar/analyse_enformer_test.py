import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('/exports/humgen/idenhond/data/DAR/Enformer_test/enformer_test.bed', names = ['chr', 'start', 'stop'], sep = '\t')
df.sort_values('chr', inplace=True)

print(f'There are {len(df)} sequences in the test set')

df['length'] = df['stop'] - df['start']

print(df)
print(df['length'].value_counts())
print(f'All sequences are 114688 bp long')

print(df['chr'].value_counts())

plt.figure()
sns.histplot(data = df, x = 'chr')
plt.xlabel('Chromosome')
plt.ylabel('Count')
plt.xticks(rotation=90)
plt.savefig('Enformer_test_hist_chr.png', bbox_inches = 'tight')
plt.close()


