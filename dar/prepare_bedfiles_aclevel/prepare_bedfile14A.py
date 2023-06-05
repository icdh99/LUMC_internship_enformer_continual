import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# change AC level cluster names to full names
df_names = pd.read_csv('/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_Ac-level_cluster.csv', sep = '\t').drop_duplicates(subset = 'Ac-level annotation', keep = False, inplace=False)
# 38 clusters ipv 43

# read supp table 14A
df = pd.read_csv('/exports/humgen/idenhond/data/DAR/AC_level/supplementary_table_14A.csv', sep = '\t')
print(df.columns)

# Cluster count
print(f'Number of DARs per cluster:')
print(df['Cluster'].value_counts())
print(f'There are {len(df["Cluster"].value_counts())} unique clusters in this table') # 43

df = pd.merge(df, df_names, how='outer', left_on='Cluster', right_on = 'Ac-level annotation')
df = df.dropna(subset=['names'])
print(f'There are {len(df["names"].value_counts())} unique clusters in this table after removing the duplicate ac-level-subclass clusters') # 38

# add columns for bed file information
df[['Chromosome', 'Start_End']] = df['Location'].str.split(':', expand=True)
df[['Start', 'End']] = df['Start_End'].str.split('-', expand=True)
df['Start'] = df['Start'].str.replace(',', '').astype(int)
df['End'] = df['End'].str.replace(',', '').astype(int)
df['Length'] = df['End'] - df['Start']
df['length_cutoff'] = df['Length'] < 256
df['FullName'] = df['names']
# length cutoff at 256 bp
print(f'There are {len(df[df["Length"] >= 256])} sequences longer than or equal to 256 bp ')

print(df)

# plot length distribution and colour sequences shorter than 256 bp
plt.figure()
sns.histplot(data = df, x = 'Length', hue = 'length_cutoff')
plt.legend(title = 'Sequence length', labels = ['< 256', '>= 256'])
plt.savefig('Supp_14A_length_cutoff.png')
plt.close()

# plot cluster counts
df_subset = df['Cluster'].value_counts().rename_axis('Cluster').reset_index(name='Counts')
plt.figure(figsize=(6.4, 8.8)) # width, heigth
sns.barplot(data = df_subset, y = 'Cluster', x = 'Counts')
plt.savefig('Supp_14A_cluster_counts.png', bbox_inches = 'tight')
plt.close()

# plot chr counts
print(df['Chromosome'].value_counts())
df_subset = df['Chromosome'].value_counts().rename_axis('Chromosomes').reset_index(name='Counts')
plt.figure()
sns.barplot(data = df_subset, y = 'Chromosomes', x = 'Counts')
plt.savefig('Supp_14A_chr_counts.png', bbox_inches = 'tight')
plt.close()

# store sequences in bed file (contains all sequences of table 14A)
bed_df = df[['Chromosome', 'Start', 'End', 'Cluster', 'FullName']]
print(f'There are {len(bed_df["Cluster"].value_counts())} unique clusters in this table') # 43
print(f'There are {len(bed_df["FullName"].value_counts())} unique clusters in this table') # 43
bed_df.to_csv('/exports/humgen/idenhond/data/DAR/AC_level/Cluster_14A.bed', sep='\t', index=False, header=False)