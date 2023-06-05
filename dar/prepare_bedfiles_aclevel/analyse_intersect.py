import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/exports/humgen/idenhond/data/DAR/AC_level/intersect_14A_test.bed', sep = '\t', names = ['chr', 'start', 'stop'])
print(df)

df['length'] = df['stop'] - df['start']
df['length_cutoff'] = df['length'] < 256
plt.figure()
sns.histplot(data = df, x = 'length', hue = 'length_cutoff')
plt.legend(title = 'Sequence length', labels = ['< 256', '>= 256'])
plt.savefig('Intersect_14A_test_lengths.png')
plt.close()

plt.figure()
sns.histplot(data = df[df['length'] >= 256], x = 'length')
plt.savefig('Intersect_14A_test_lengths_above256.png')
plt.close()

df = df[df['length'] >= 256]
print(df)

print(df['chr'].value_counts())
df_subset = df['chr'].value_counts().rename_axis('Chr').reset_index(name='Counts')
plt.figure()
sns.barplot(data = df_subset, y = 'Chr', x = 'Counts')
plt.savefig('Intersect_14A_test_chr_counts.png', bbox_inches = 'tight')
plt.close()

print(f'There are {len(df[df["length"] >= 256])} sequences equal to or longer than 256 bp in the intersect')