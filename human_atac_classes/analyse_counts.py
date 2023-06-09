import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## load correlation on the test set for these tracks
df_corr_test = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_humanatac.csv')
df_corr_test = df_corr_test.rename(columns = {'Unnamed: 0' : 'Index old'})
df_corr_valid = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_humanatac.csv')
df_corr_valid = df_corr_valid.rename(columns = {'Unnamed: 0' : 'Index old'})
df_corr_train = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train_humanatac.csv')
df_corr_train = df_corr_train.rename(columns = {'Unnamed: 0' : 'Index old'})

input_file = '/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_classes_counts.csv'
df = pd.read_csv(input_file, sep = ',').drop_duplicates(subset = 'Index old', keep = False, inplace=False)
print(df)
# idx_subclass_ac = [35, 29,  30, 36, 37, 39, 40, 33, 25, 38, 41, 42, 31, 27, 28, 32, 8, 9, 10, 5, 6, 11, 12, 13, 19, 17, 15, 16, 21, 18, 20, 3, 1, 2, 52, 56, 57, 59]
# df = df.loc[df['Index old'].isin(idx_subclass_ac)] #.sort_values(by = 'names')

df_all = df.merge(df_corr_test, left_on='Index old', right_on = 'Index old').rename(columns = {'0' : 'Test correlation'})
df_all = df_all.merge(df_corr_valid, left_on='Index old', right_on = 'Index old').rename(columns = {'0' : 'Valid correlation'})
df_all = df_all.merge(df_corr_train, left_on='Index old', right_on = 'Index old').rename(columns = {'0' : 'Train correlation'})
df_all = df_all.drop(['file', 'clip', 'sum_stat', 'scale'], axis = 1)
print(df_all)
print(df_all.shape)
print(df_all.columns)

print(f'Mean correlation Test: {df_all["Test correlation"].mean()}')
print(f'Mean correlation Valid: {df_all["Valid correlation"].mean()}')
print(f'Mean correlation Train: {df_all["Train correlation"].mean()}')

df_class = df[df['level'] == 'Class'].reset_index().drop(labels=['index'], axis = 'columns')
df_class = df_class.merge(df_corr_test, left_on='Index old', right_on = 'Index old').rename(columns = {'0' : 'Test correlation'})
df_class = df_class.merge(df_corr_valid, left_on='Index old', right_on = 'Index old').rename(columns = {'0' : 'Valid correlation'})
df_class = df_class.merge(df_corr_train, left_on='Index old', right_on = 'Index old').rename(columns = {'0' : 'Train correlation'})
# print(df_class)

df_subclass = df[df['level'] == 'Subclass'].reset_index().drop(labels=['index'], axis = 'columns').sort_values(by = 'Class', axis = 'index')
df_subclass = df_subclass.merge(df_corr_test, left_on='Index old', right_on = 'Index old').rename(columns = {'0' : 'Test correlation'})
df_subclass = df_subclass.merge(df_corr_valid, left_on='Index old', right_on = 'Index old').rename(columns = {'0' : 'Valid correlation'})
df_subclass = df_subclass.merge(df_corr_train, left_on='Index old', right_on = 'Index old').rename(columns = {'0' : 'Train correlation'})
# print(df_subclass)
# print(df_subclass[['names']])

df_aclevel = df[df['level'] == 'Ac-level cluster'].drop(labels=['index'], axis = 'columns').sort_values(by = 'Class', axis = 'index')
df_aclevel['subclass+aclevel'] = df_aclevel['Subclass'] + ' ' + df_aclevel['Ac-level annotation']
df_aclevel = df_aclevel.merge(df_corr_test, left_on='Index old', right_on = 'Index old').rename(columns = {'0' : 'Test correlation'})
df_aclevel = df_aclevel.merge(df_corr_valid, left_on='Index old', right_on = 'Index old').rename(columns = {'0' : 'Valid correlation'})
df_aclevel = df_aclevel.merge(df_corr_train, left_on='Index old', right_on = 'Index old').rename(columns = {'0' : 'Train correlation'})
# print(df_aclevel)

print(f'Total number of cells at Class level: {df_class["Nuclei"].sum()}')
print(f'Total number of cells at Sublass level: {df_subclass["Nuclei"].sum()}')
print(f'Total number of cells at AC-cluster level: {df_aclevel["Nuclei"].sum()}')
print(f'Mean correlation Test at Class level: {df_class["Test correlation"].mean():.3f}')
print(f'Mean correlation Test at Sublass level: {df_subclass["Test correlation"].mean():.3f}')
print(f'Mean correlation Test at AC-cluster level: {df_aclevel["Test correlation"].mean():.3f}')
print(f'Mean correlation Valid at Class level: {df_class["Valid correlation"].mean():.3f}')
print(f'Mean correlation Valid at Sublass level: {df_subclass["Valid correlation"].mean():.3f}')
print(f'Mean correlation Valid at AC-cluster level: {df_aclevel["Valid correlation"].mean():.3f}')
print(f'Mean correlation Train at Class level: {df_class["Train correlation"].mean():.3f}')
print(f'Mean correlation Train at Sublass level: {df_subclass["Train correlation"].mean():.3f}')
print(f'Mean correlation Train at AC-cluster level: {df_aclevel["Train correlation"].mean():.3f}')

print(df_all)
value_mapping = {'Class': 1, 'Subclass': 2, 'Ac-level cluster': 3}
df_all['Sort'] = df_all['level'].map(value_mapping)
df_all = df_all.sort_values(by = 'Sort', axis = 'index')
print(df_all)

# groups = df_all.groupby('Sort')
# fig, axs = plt.subplots(nrows=1, ncols=len(groups), sharex=False, sharey=True, width_ratios=[1, 6, 12], constrained_layout = True, figsize = (6.4*2, 4.8*2)) #
# for (name, group), ax1 in zip(groups, axs):
#     print(name) # 1, 2, 3
#     # ax1.grid()
#     ax1.set_axisbelow(True)
#     # ax1.grid(axis='both', which='both')
#     ax1.yaxis.grid(color='gray')
#     # ax1.xaxis.grid(True, color='gray', linestyle='dashed')

#     group = group.sort_values(by = 'Class', axis = 'index')
#     print(group.shape)
#     ax1.tick_params(axis='x', labelrotation = 90)
#     ax2 = ax1.twinx()
#     ax2.set_ylim([0, 1])
#     ax1.set_xlabel(None)
#     ax2.set_xlabel(None)

#     print(group)
#     sns.barplot(data = group, x = 'names', y = 'Nuclei', ax = ax1, hue = 'Class', dodge=False)
#     sns.lineplot(data = group, x = 'names', y = 'Test correlation', ax=ax2, marker = 'o', color = 'k', linestyle='')
#     if name == 1 or name == 2:
#         ax2.set_yticklabels([]) 
#         ax2.set_ylabel(None)
#     if name == 2 or name == 3: 
#         ax1.set_ylabel(None)
    
# plt.savefig('/exports/humgen/idenhond/projects/human_atac_classes/Plots/barplot_test.png', bbox_inches='tight', dpi = 300)
# plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig3_ATAC/barplot_test.png', bbox_inches='tight', dpi = 300)

# fig, axs = plt.subplots(nrows=1, ncols=len(groups), sharex=False, sharey=True, width_ratios=[1, 6, 12], constrained_layout = True, figsize = (6.4*2, 4.8*2)) #
# for (name, group), ax1 in zip(groups, axs):
#     group = group.sort_values(by = 'Class', axis = 'index')
#     ax1.tick_params(axis='x', labelrotation = 90)
#     ax2 = ax1.twinx()
#     ax2.set_ylim([0, 1])
#     ax1.set_xlabel(None)
#     ax2.set_xlabel(None)
#     sns.barplot(data = group, x = 'names', y = 'Nuclei', ax = ax1, hue = 'Class', dodge=False)
#     sns.lineplot(data = group, x = 'names', y = 'Train correlation', ax=ax2, marker = 'o', color = 'k', linestyle='')
#     if name == 1 or name == 2:
#         ax2.set_yticklabels([]) 
#         ax2.set_ylabel(None)
#     if name == 2 or name == 3: 
#         ax1.set_ylabel(None)
# plt.savefig('/exports/humgen/idenhond/projects/human_atac_classes/Plots/barplot_train.png', bbox_inches='tight', dpi = 300)
# plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig3_ATAC/barplot_train.png', bbox_inches='tight', dpi = 300)

# fig, axs = plt.subplots(nrows=1, ncols=len(groups), sharex=False, sharey=True, width_ratios=[1, 6, 12], constrained_layout = True, figsize = (6.4*2, 4.8*2)) #
# for (name, group), ax1 in zip(groups, axs):
#     group = group.sort_values(by = 'Class', axis = 'index')
#     ax1.tick_params(axis='x', labelrotation = 90)
#     ax2 = ax1.twinx()
#     ax2.set_ylim([0, 1])
#     ax1.set_xlabel(None)
#     ax2.set_xlabel(None)
#     sns.barplot(data = group, x = 'names', y = 'Nuclei', ax = ax1, hue = 'Class', dodge=False)
#     sns.lineplot(data = group, x = 'names', y = 'Valid correlation', ax=ax2, marker = 'o', color = 'k', linestyle='')
#     if name == 1 or name == 2:
#         ax2.set_yticklabels([]) 
#         ax2.set_ylabel(None)
#     if name == 2 or name == 3: 
#         ax1.set_ylabel(None)
# plt.savefig('/exports/humgen/idenhond/projects/human_atac_classes/Plots/barplot_valid.png', bbox_inches='tight', dpi = 300)
# plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig3_ATAC/barplot_valid.png', bbox_inches='tight', dpi = 300)

plt.figure()
plt.tight_layout()
plt.grid(True)
ax = sns.scatterplot(data = df_all, x = 'Nuclei', y = 'Test correlation', s = 50, color = 'k')
plt.ylim(0,1)
plt.xticks([0, 10000, 20000, 30000, 40000], [0, 10000, 20000, 30000, 40000])
plt.savefig('/exports/humgen/idenhond/projects/human_atac_classes/Plots/scatterplot_allclusters.png', bbox_inches='tight', dpi = 300)
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig3_ATAC/scatterplot_allclusters.png', bbox_inches='tight', dpi = 300)
plt.close()

exit()
fig, ax1 = plt.subplots(figsize=(12,8))
sns.barplot(data = df_class, x = 'Class', y = 'Nuclei')
ax2 = ax1.twinx()
ax2.set_ylim([0, 1])
sns.lineplot(data = df_class, x = 'Class', y = 'Test correlation', ax=ax2, marker = 'o', color = 'k', linestyle='')
for pos, row in df_class.iterrows():
    ax2.annotate(f"{row['Test correlation']:.3f}", (pos, row['Test correlation']*1.06), color = 'k', va='top', ha='center')
plt.savefig('/exports/humgen/idenhond/projects/human_atac_classes/Plots/barplot_class.png', bbox_inches='tight')
plt.close()

fig, ax1 = plt.subplots(figsize=(12,8))
sns.barplot(data = df_subclass, x = 'Subclass', y = 'Nuclei', hue = 'Class', ax = ax1, alpha = 0.7)
ax2 = ax1.twinx()
ax2.set_ylim([0, 1])
sns.lineplot(data = df_subclass, x = 'Subclass', y = 'Test correlation', ax=ax2, marker = 'o', color = 'k', linestyle='')
ax1.tick_params(axis='x', labelrotation = 90)
# plt.xticks(rotation = 45, ha = 'right')
# ax1.set_xticklabels(ax1.get_xticks(), rotation = 45)
# ax1.tick_params(axis='x', labelrotation = 45, ha = 'right')
plt.savefig('/exports/humgen/idenhond/projects/human_atac_classes/Plots/barplot_subclass.png', bbox_inches='tight')
plt.close()

fig, ax1 = plt.subplots(figsize=(12,8))

# df_aclevel_sorted = df_aclevel.sort_values(['Class', 'Nuclei'])
# sns.barplot(data = df_aclevel, x = 'Ac-level annotation', y = 'Nuclei') # 5 bars met 2 values
sns.barplot(data = df_aclevel, x = 'subclass+aclevel', y = 'Nuclei', hue = 'Class', ax = ax1, alpha = 0.7) 
ax2 = ax1.twinx()
ax2.set_ylim([0, 1])
sns.lineplot(data = df_aclevel, x = 'subclass+aclevel', y = 'Test correlation', ax=ax2, marker = 'o', color = 'k', linestyle='')
ax1.tick_params(axis='x', labelrotation = 90)
# plt.xticks(rotation = 90, ha = 'left')
# ax1.set_xticklabels(ax1.get_xticks(), rotation = 45)
# ax1.tick_params(axis='x', labelrotation = 45)
# ax1.set_xticklabels(ax1.get_xticks(), ha = 'right')
plt.savefig('/exports/humgen/idenhond/projects/human_atac_classes/Plots/barplot_aclevel.png', bbox_inches='tight')
plt.close()

plt.figure()
sns.scatterplot(data = df_class, x = 'Nuclei', y = 'Test correlation', hue = 'Class')
plt.ylim(0,1)
plt.title('Class clusters')
plt.savefig('/exports/humgen/idenhond/projects/human_atac_classes/Plots/scatterplot_class.png', bbox_inches='tight')
plt.close()

plt.figure()
sns.scatterplot(data = df_subclass, x = 'Nuclei', y = 'Test correlation', hue = 'Class')
plt.ylim(0,1)
plt.xscale('log')
plt.title('Subclass clusters')
plt.savefig('/exports/humgen/idenhond/projects/human_atac_classes/Plots/scatterplot_subclass.png', bbox_inches='tight')
plt.close()

plt.figure()
sns.scatterplot(data = df_aclevel, x = 'Nuclei', y = 'Test correlation', hue = 'Class')
plt.ylim(0,1)
plt.xscale('log')
plt.title('Ac level clusters')
plt.savefig('/exports/humgen/idenhond/projects/human_atac_classes/Plots/scatterplot_aclevel.png', bbox_inches='tight')
plt.close()

print(df_all['Class'])
print(df_all['Subclass'])
df_all['Class + Subclass'] = df_all['Class'] + ' ' + df_all['Subclass']
# print(df_all['Class + Subclass'])
plt.figure()
sns.scatterplot(data = df_all, x = 'Nuclei', y = 'Test correlation', hue = 'level')
plt.ylim(0,1)
# plt.xscale('log')
plt.title('All clusters')
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.savefig('/exports/humgen/idenhond/projects/human_atac_classes/Plots/scatterplot_allclusters.png', bbox_inches='tight', dpi = 300)
plt.close()

print(df_aclevel['Nuclei'].sum())
print(df_subclass['Nuclei'].sum())
print(df_class['Nuclei'].sum())









# plt.figure()
# df_aclevel.plot.pie(y = 'Nuclei')
# plt.savefig('/exports/humgen/idenhond/projects/human_atac_classes/Plots/pieplot_aclevel.png', bbox_inches='tight')
# plt.close

# plt.figure()
# df_subclass.plot.pie(y = 'Nuclei')
# plt.savefig('/exports/humgen/idenhond/projects/human_atac_classes/Plots/pieplot_subclass.png', bbox_inches='tight')
# plt.close

# plt.figure()
# df_class.plot.pie(y = 'Nuclei')
# plt.savefig('/exports/humgen/idenhond/projects/human_atac_classes/Plots/pieplot_class.png', bbox_inches='tight')
# plt.close