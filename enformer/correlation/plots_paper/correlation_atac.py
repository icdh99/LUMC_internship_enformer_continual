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

input_file_subclass = '/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_Subclass.csv'
df_subclass = pd.read_csv(input_file_subclass, sep = '\t').rename(columns = {'Unnamed: 0' : 'Index per level'})
df_subclass_test = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_humanatac_subclass.csv',index_col = 0).rename(columns = {'0' : 'Test correlation'})
df_subclass_valid = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_humanatac_subclass.csv',index_col = 0).rename(columns = {'0' : 'Valid correlation'})
df_subclass_train = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train_humanatac_subclass.csv',index_col = 0).rename(columns = {'0' : 'Train correlation'})
df_subclass['Test correlation'] = df_subclass_test['Test correlation']
df_subclass['Valid correlation'] = df_subclass_valid['Valid correlation']
df_subclass['Train correlation'] = df_subclass_train['Train correlation']

input_file_class = '/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_Class.csv'
df_class = pd.read_csv(input_file_class, sep = '\t').rename(columns = {'Unnamed: 0' : 'Index per level'})
df_class_test = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_humanatac_class.csv',index_col = 0).rename(columns = {'0' : 'Test correlation'})
df_class_valid = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_humanatac_class.csv',index_col = 0).rename(columns = {'0' : 'Valid correlation'})
df_class_train = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train_humanatac_class.csv',index_col = 0).rename(columns = {'0' : 'Train correlation'})
df_class['Test correlation'] = df_class_test['Test correlation']
df_class['Valid correlation'] = df_class_valid['Valid correlation']
df_class['Train correlation'] = df_class_train['Train correlation']

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
df = df.merge(df_correlation_train, left_on='Index old', right_on = 'Index old')
print(df.columns)
print(df[['Test correlation', 'level']])


print(f'mean correlation score Test: {df["Test correlation All tracks"].mean(axis=0):.4f}')
print(f'mean correlation score Test per class: {df["Test correlation"].mean(axis=0):.4f}')
print(f'mean correlation score Test class: {df[df["level"] == "Class"]["Test correlation All tracks"].mean(axis=0):.4f}')
print(f'mean correlation score Test subclass: {df[df["level"] == "Subclass"]["Test correlation All tracks"].mean(axis=0):.4f}')
print(f'mean correlation score Test ac level: {df[df["level"] == "Ac-level cluster"]["Test correlation All tracks"].mean(axis=0):.4f}')
print(f'mean correlation score Validation: {df["Validation correlation All tracks"].mean(axis=0):.4f}')
print(f'mean correlation score Train: {df["Train correlation All tracks"].mean(axis=0):.4f}')


exit()
# plot correlation of old model (66 tracks) vs new models (trained per level)
# plt.figure()
# plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
# sns.scatterplot(data = df, x = 'Test correlation All tracks', y = 'Test correlation', hue = 'level')
# plt.xlabel('Model trained on all pseudo bulk cell type profiles')
# plt.ylabel('Models trained on pseudo bulk cell type profiles ')
# plt.title('Test set')
# plt.legend(title = 'Model level')
# plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig3_ATAC/atac_perlevel_scatterplot_test.png', bbox_inches='tight', dpi = 300)
# plt.close()

# plt.figure()
# plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
# sns.scatterplot(data = df, x = 'Validation correlation All tracks', y = 'Valid correlation', hue = 'level')
# plt.xlabel('Model trained on all pseudo bulk cell type profiles')
# plt.ylabel('Models trained on one level of pseudo bulk cell type profiles ')
# plt.title('Validation set')
# plt.legend(title = 'Model level')
# plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig3_ATAC/atac_perlevel_scatterplot_valid.png', bbox_inches='tight', dpi = 300)
# plt.close()

plt.figure()
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df, x = 'Validation correlation All tracks', y = 'Valid correlation', hue = 'level')
plt.xlabel('Model trained on all pseudo bulk cell type profiles')
plt.ylabel('Models trained on pseudo bulk cell type profiles ')
plt.title('Train set')
plt.legend(title = 'Model level')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig3_ATAC/atac_perlevel_scatterplot_train.png', bbox_inches='tight', dpi = 300)
plt.close()


fig, (ax1, ax2, ax3) = plt.subplots(1, ncols = 3, sharex=True, sharey=True, constrained_layout=True, figsize=(10, 4.8))
sns.despine(top=True, right=True, left=False, bottom=False)
ax1.set_aspect('equal', adjustable='box')
ax2.set_aspect('equal', adjustable='box')
ax3.set_aspect('equal', adjustable='box')

ax1.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
ax2.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
ax3.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')

sns.scatterplot(data = df, x = 'Test correlation All tracks', y = 'Test correlation', hue = 'level', ax = ax1)
sns.scatterplot(data = df, x = 'Validation correlation All tracks', y = 'Valid correlation', hue = 'level', ax = ax2)
sns.scatterplot(data = df, x = 'Train correlation All tracks', y = 'Train correlation', hue = 'level', ax = ax3)

ax1.set_xlabel(None)
ax1.set_ylabel(None)
ax2.set_xlabel(None)
ax2.set_ylabel(None)
ax3.set_xlabel(None)
ax3.set_ylabel(None)

ax1.get_legend().remove()
ax2.get_legend().remove()

ax1.title.set_text('Test set')
ax2.title.set_text('Validation set')
ax3.title.set_text('Train set')

ax1.text(0.9, 0.03, '0.531', fontsize = 8, ha='center', va='center', transform=ax1.transAxes)
ax2.text(0.9, 0.03, '0.493', fontsize = 8, ha='center', va='center', transform=ax2.transAxes)
ax3.text(0.9, 0.03, '0.551', fontsize = 8, ha='center', va='center', transform=ax3.transAxes)

ax3.legend(loc = 'upper left', bbox_to_anchor=(1.1, 1.05))
ax3.get_legend().set_title('Level')

# fig.supxlabel('Model trained on all pseudo bulk cell type profiles')
plt.figtext(.5, .17, 'Model trained on all pseudo bulk cell type profiles', fontsize=9, ha='center')
fig.supylabel(f'   Models trained on one level \nof pseudo bulk cell type profiles', fontsize = 9)
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig3_ATAC/atac_perlevel_scatterplot.png', bbox_inches='tight', dpi = 300)
