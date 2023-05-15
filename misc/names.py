import pandas as pd

# df = pd.read_csv('/exports/humgen/idenhond/ac-level_cluster.csv')
# print(df)
# print(df.nunique())

# df = pd.read_csv('/exports/humgen/idenhond/subclass.csv')
# print(df)
# print(df.nunique())

# df = pd.read_csv('/exports/humgen/idenhond/class.csv')
# print(df)
# print(df.nunique())


df = pd.read_csv('targets_classes.csv', index_col='index', sep = ';')
# df = df.sort_values(by = ['index'])
print(df)
print(df.columns)
cols = ['Class', 'Subclass', 'Ac-level annotation']
df['names'] = df[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)
print(df)
