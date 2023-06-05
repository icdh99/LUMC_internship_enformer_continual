import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/exports/humgen/idenhond/data/DAR/Subclass_level/supplementary_table_14C.csv', sep = '\t')
print(df)

print(f'Number of DARs per cluster:')
print(df['Subclass'].value_counts())
df_subset = df['Subclass'].value_counts().rename_axis('Subclass').reset_index(name='Counts')
print(df_subset)
plt.figure()
sns.barplot(data = df_subset, y = 'Subclass', x = 'Counts')
plt.savefig('Supp_14C_subclass_counts.png', bbox_inches = 'tight')

print(f'There are {len(df["Subclass"].value_counts())} unique clusters in this table') # 20


# print(list(set(list(df['Subclass']))))
subclasses = ['Micro-PVM', 'SNCG', 'L2-3 IT', 'LAMP5', 'L6b', 'L5 ET', 'SST', 'L5-6 NP', 'L6 CT', 'VLMC', 'Oligo', 'L6 IT Car3', 'PVALB', 'VIP', 'Astro', 'Endo', 'L6 IT', 'OPC', 'SST CHODL', 'L5 IT']
print(f'There are {len(subclasses)} subclasses')
subclasses_neuronal = [ 'SNCG', 'L2-3 IT', 'LAMP5', 'L6b', 'L5 ET', 'SST', 'L5-6 NP', 'L6 CT',   'L6 IT Car3', 'PVALB', 'VIP',  'L6 IT',  'SST CHODL', 'L5 IT']
print(f'There are {len(subclasses_neuronal)} neuronal subclasses')

df_neuronal = df[df['Subclass'].isin(subclasses_neuronal)]
print(df_neuronal)
df_subset = df_neuronal['Subclass'].value_counts().rename_axis('Subclass').reset_index(name='Counts')
print(df_subset)
plt.figure()
sns.barplot(data = df_subset, y = 'Subclass', x = 'Counts')
plt.savefig('Supp_14C_subclass_neuronal_counts.png', bbox_inches = 'tight')

# for cluster, data in grouped:
#     print(type(cluster))
#     cluster = cluster.replace(' ', '_')
#     print(cluster)
#     # print(data) # df
#     filename = f'/exports/humgen/idenhond/data/DAR/AC_level/Cluster_{cluster}.bed'
#     data['Location'] = data['Location'].str.split(':|-').apply(lambda x: '\t'.join(x[0:3]))
#     data[['Location']].to_csv(filename, sep='\t', index=False)
#     break
