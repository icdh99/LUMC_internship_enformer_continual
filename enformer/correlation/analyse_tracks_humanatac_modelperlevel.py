import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_file_aclevel = '/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_Ac-level_cluster.csv'
df_aclevel = pd.read_csv(input_file_aclevel, sep = '\t').rename(columns = {'Unnamed: 0' : 'Index per level'})
df_aclevel_test = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_humanatac_aclevel.csv',index_col = 0).rename(columns = {'0' : 'Test correlation'})
df_aclevel_valid = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_humanatac_aclevel.csv',index_col = 0).rename(columns = {'0' : 'Valid correlation'})
df_aclevel_train = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train_humanatac_aclevel.csv',index_col = 0).rename(columns = {'0' : 'Train correlation'})
df_aclevel['Test correlation'] = df_aclevel_test['Test correlation']
df_aclevel['Valid correlation'] = df_aclevel_valid['Valid correlation']
df_aclevel['Train correlation'] = df_aclevel_train['Train correlation']
# print(df_aclevel)

input_file_subclass = '/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_Subclass.csv'
df_subclass = pd.read_csv(input_file_subclass, sep = '\t').rename(columns = {'Unnamed: 0' : 'Index per level'})
df_subclass_test = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_humanatac_subclass.csv',index_col = 0).rename(columns = {'0' : 'Test correlation'})
df_subclass_valid = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_humanatac_subclass.csv',index_col = 0).rename(columns = {'0' : 'Valid correlation'})
df_subclass_train = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train_humanatac_subclass.csv',index_col = 0).rename(columns = {'0' : 'Train correlation'})
df_subclass['Test correlation'] = df_subclass_test['Test correlation']
df_subclass['Valid correlation'] = df_subclass_valid['Valid correlation']
df_subclass['Train correlation'] = df_subclass_train['Train correlation']
# print(df_subclass)

input_file_class = '/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_Class.csv'
df_class = pd.read_csv(input_file_class, sep = '\t').rename(columns = {'Unnamed: 0' : 'Index per level'})
df_class_test = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_humanatac_class.csv',index_col = 0).rename(columns = {'0' : 'Test correlation'})
df_class_valid = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_humanatac_class.csv',index_col = 0).rename(columns = {'0' : 'Valid correlation'})
df_class_train = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train_humanatac_class.csv',index_col = 0).rename(columns = {'0' : 'Train correlation'})
df_class['Test correlation'] = df_class_test['Test correlation']
df_class['Valid correlation'] = df_class_valid['Valid correlation']
df_class['Train correlation'] = df_class_train['Train correlation']
# print(df_class)

#concat all dataframes
print(f'Number of tracks ac level: {df_aclevel.shape[0]}')
print(f'Number of tracks subclass: {df_subclass.shape[0]}')
print(f'Number of tracks class: {df_class.shape[0]}')
df = pd.concat([df_class, df_subclass, df_aclevel], ignore_index=True)

# test and validation and train correlation of model trained on 66 human atac seq tracks
df_correlation_test = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_humanatac.csv').tail(-1).rename(columns = {'Unnamed: 0' : 'Index old', '0' : 'Test correlation All tracks'})
df_correlation_valid = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_humanatac.csv').tail(-1).rename(columns = {'Unnamed: 0' : 'Index old', '0' : 'Validation correlation All tracks'})
df_correlation_train = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train_humanatac.csv').tail(-1).rename(columns = {'Unnamed: 0' : 'Index old', '0' : 'Train correlation All tracks'})

df = df.merge(df_correlation_test, left_on='Index old', right_on = 'Index old')
df = df.merge(df_correlation_valid, left_on='Index old', right_on = 'Index old')
print(df.columns)
print(df)

# plot correlation of old model (66 tracks) vs new models (trained per level)
plt.figure()
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df, x = 'Test correlation All tracks', y = 'Test correlation', hue = 'level')
plt.xlabel('Model trained on all ATAC-seq tracks (66)')
plt.ylabel('Models trained on one level of ATAC-seq tracks')
plt.title('Pearson correlation coefficient for test set - Human ATAC-seq tracks')
plt.legend(title = 'Model level')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/human_atac_perlevel/atac_perlevel_scatterplot_test.png', bbox_inches='tight')
plt.close()

plt.figure()
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df, x = 'Validation correlation All tracks', y = 'Valid correlation', hue = 'level')
plt.xlabel('Model trained on all ATAC-seq tracks (66)')
plt.ylabel('Models trained on one level of ATAC-seq tracks')
plt.title('Pearson correlation coefficient for validation set - Human ATAC-seq tracks')
plt.legend(title = 'Model level')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/human_atac_perlevel/atac_perlevel_scatterplot_valid.png', bbox_inches='tight')
plt.close()

plt.figure()
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df, x = 'Validation correlation All tracks', y = 'Valid correlation', hue = 'level')
plt.xlabel('Model trained on all ATAC-seq tracks (66)')
plt.ylabel('Models trained on one level of ATAC-seq tracks')
plt.title('Pearson correlation coefficient for validation set - Human ATAC-seq tracks')
plt.legend(title = 'Model level')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/human_atac_perlevel/atac_perlevel_scatterplot_train.png', bbox_inches='tight')
plt.close()