import pandas as pd

# df = pd.read_csv('/exports/humgen/idenhond/targets_classes.csv', index_col='index', sep = ';')

# cols = ['Class', 'Subclass', 'Ac-level annotation']
# df['names'] = df[cols].apply(lambda row: ' '.join(row.values.astype(str)), axis=1)

# # print(df[['identifier', 'Class', 'Subclass', 'Ac-level annotation']])
# print(df[['identifier', 'Class', 'Subclass', 'Ac-level annotation']].to_markdown())

# df.to_csv('/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_classes.csv', sep = '\t')

df = pd.read_csv('/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_classes.csv', index_col='index', sep = '\t')
print(df)

#hierna heb ik met de hand de ac-level clusters met een dubbele annotatie twee keer toegevoegd, 1x per subclass. gecommente code hierboven niet meer runnen
# vijf extra rijen 

print(df.columns)

# maak een targets file voor elk level (class, subclass, ac level annotation)

print(df['level'].value_counts())

df_class = df[df['level'] == 'Class'].reset_index().drop(labels=['index'], axis = 'columns')
print(df_class)
df_class.to_csv('/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_Class.csv', sep = '\t', index = True)
print(list(df_class['Index old']))

df_subclass = df[df['level'] == 'Subclass'].reset_index().drop(labels=['index'], axis = 'columns')
# print(df_subclass)
df_subclass.to_csv('/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_Subclass.csv', sep = '\t')
print(list(df_subclass['Index old']))
print(len(list(df_subclass['Index old'])))

df_aclevel = df[df['level'] == 'Ac-level cluster'].reset_index().drop(labels=['index'], axis = 'columns')
# print(df_aclevel)
df_aclevel.to_csv('/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_Ac-level_cluster.csv', sep = '\t')
print(len(list(df_aclevel['Index old'])))
print(list(df_aclevel['Index old']))
