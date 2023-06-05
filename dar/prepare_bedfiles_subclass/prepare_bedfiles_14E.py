import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/exports/humgen/idenhond/data/DAR/Subclass_level/supplementary_table_14E.csv', sep = '\t')
print(df)

print(f'Number of DARs per cluster:')
print(df['Subclass'].value_counts())
print(f'There are {len(df["Subclass"].value_counts())} unique clusters in this table') # 14

df_subset = df['Subclass'].value_counts().rename_axis('Subclass').reset_index(name='Counts')
plt.figure()
sns.barplot(data = df_subset, y = 'Subclass', x = 'Counts')
plt.savefig('Supp_14E_subclass_counts.png', bbox_inches = 'tight')


df[['Chromosome', 'Start_End']] = df['Location'].str.split(':', expand=True)
df[['Start', 'End']] = df['Start_End'].str.split('-', expand=True)

bed_df = df[['Chromosome', 'Start', 'End', 'Subclass']]
bed_df['Start'] = bed_df['Start'].str.replace(',', '').astype(int)
bed_df['End'] = bed_df['End'].str.replace(',', '').astype(int)
bed_df.to_csv('/exports/humgen/idenhond/data/DAR/Subclass_level/Subclass_neuronal_14E.bed', sep='\t', index=False, header=False)

print(bed_df)
df_subset = bed_df['Chromosome'].value_counts().rename_axis('Chr').reset_index(name='Counts')
plt.figure()
sns.barplot(data = df_subset, y = 'Chr', x = 'Counts')
plt.savefig('Supp_14E_chr_counts.png', bbox_inches = 'tight')

