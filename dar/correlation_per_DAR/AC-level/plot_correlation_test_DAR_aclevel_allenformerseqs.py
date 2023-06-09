import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

random.seed(18)

idx_subclass = [35, 29,  30, 36, 37, 39, 40, 33, 25, 38, 41, 42, 31, 27, 28, 32, 8, 9, 10, 5, 6, 11, 12, 13, 19, 17, 15, 16, 21, 18, 20, 3, 1, 2, 52, 56, 57, 59] #38
df_ac = pd.read_csv('/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_Ac-level_cluster.csv', sep = '\t').drop_duplicates(subset = 'Index old', keep = 'first', inplace=False)
df_ac = df_ac.loc[df_ac['Index old'].isin(idx_subclass)] #.sort_values(by = 'names')
idx_subclass_sorted = df_ac['Index old'].to_list()
ac_labels = df_ac['names'].to_list()
print(len(ac_labels))
df_counts = pd.read_csv('/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_classes_counts.csv')
df_ac = df_ac.merge(df_counts[['Index old', 'Nuclei']], on = 'Index old')
print(df_ac)

counts = df_ac['Nuclei'].to_numpy()
print(counts)

df_withnames = pd.read_csv('/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_Ac-level_cluster.csv', sep = '\t')
print(df_withnames)

df_correlations = pd.read_csv('/exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/Correlation_DAR_aclevel.csv', index_col = 'Unnamed: 0')
df_correlations['Index1'] = df_correlations.index
print(df_correlations)

df_correlations_subclass = pd.read_csv('/exports/humgen/idenhond/projects/dar/correlation_per_DAR/Subclass/Correlation_DAR_subclass.csv', index_col = 'Unnamed: 0')
df_correlations_subclass['Index1'] = df_correlations_subclass.index
print(df_correlations_subclass)

df_correlations['Source'] = 'AC-level'
df_correlations_subclass['Source'] = 'Subclass'

combined_df = pd.concat([df_correlations[['Correlation', 'Source']], df_correlations_subclass[['Correlation', 'Source']]])
combined_df = combined_df.reset_index(drop=True)

print(combined_df)

plt.figure()
ax = sns.histplot(data=df_correlations, x="Correlation", color = 'k', element = 'step')
plt.xlabel('Pearson correlation coefficient', fontsize=9)
plt.ylabel('Count', fontsize=9)
ax.tick_params(axis='both', which='major', labelsize=8)
plt.savefig('Plots/Correlation_DAR_aclevel.png', bbox_inches = 'tight', dpi = 300)
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig4_DAR/Correlation_DAR_aclevel.png', bbox_inches = 'tight', dpi = 300)
plt.close()

plt.figure()
sns.histplot(data=combined_df, x="Correlation", hue = 'Source', hue_order=['AC-level','Subclass'])
plt.xlabel('Pearson correlation coefficient')
plt.savefig('Plots/Correlation_DAR_combined.png', bbox_inches = 'tight', dpi = 300)
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig4_DAR/Correlation_DAR_combined.png', bbox_inches = 'tight', dpi = 300)
plt.close()

df_correlations = df_correlations.sort_values(by = 'Correlation')
print(df_correlations)

print(f"mean: {df_correlations['Correlation'].mean()}")
print(f"median: {df_correlations['Correlation'].median()}")

pred = np.loadtxt('/exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/Predictions_DAR_aclevel.csv',delimiter=',')
print(pred.shape)  

print(pred.min())
print(pred.max())

target = np.loadtxt('/exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/Targets_DAR_aclevel.csv' ,delimiter=',')
print(target.shape)

fig, ax = plt.subplots()
plt.imshow(pred, interpolation='none', cmap='viridis', origin = 'lower', extent = [0, 357690, 0, 38], aspect=9500)
ax.set_yticks(np.arange(len(ac_labels)) + 0.5, ac_labels)
ax.set_yticklabels([], va='center', fontsize=4)
ax.xaxis.set_tick_params(labelsize=3)
cbar = plt.colorbar(shrink=0.3, orientation = "horizontal", location = 'top', fraction=0.026, pad=0.06)
cbar.outline.set_visible(False)
cbar.set_ticks([0, 5])
cbar.set_ticklabels([0, 5])
cbar.ax.tick_params(size=0, labelsize = 4)
cbar.ax.xaxis.set_ticks_position('bottom')
cbar.ax.xaxis.set_label_position('bottom')
plt.xlabel('DARs', fontsize = 4)
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig4_DAR/Plot_Intersect_PredictedValues_allseqs.png', bbox_inches='tight', dpi = 800)
plt.savefig('Plot_Intersect_PredictedValues_allseqs.png', bbox_inches='tight', dpi = 800)

fig, ax = plt.subplots()
plt.imshow(target, interpolation='none', cmap='viridis', origin = 'lower', extent = [0, 357690, 0, 38], aspect=9500)
ax.set_yticks(np.arange(len(ac_labels)) + 0.5, ac_labels)
ax.set_yticklabels(ac_labels, va='center', fontsize=4)
ax.xaxis.set_tick_params(labelsize=3)
cbar = plt.colorbar(shrink=0.3, orientation = "horizontal", location = 'top', fraction=0.026, pad=0.06)
cbar.outline.set_visible(False)
cbar.set_ticks([0, 60])
cbar.set_ticklabels([0, 60])
cbar.ax.tick_params(size=0, labelsize = 4)
cbar.ax.xaxis.set_ticks_position('bottom')
cbar.ax.xaxis.set_label_position('bottom')
plt.xlabel('DARs', fontsize = 4)
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig4_DAR/Plot_Intersect_TargetValues_allseqs.png', bbox_inches='tight', dpi = 800)
plt.savefig('Plot_Intersect_TargetValues_allseqs.png', bbox_inches='tight', dpi = 800)


# pred = np.divide(pred.T,counts).T
# fig, ax = plt.subplots()
# plt.imshow(pred, interpolation='none', cmap='viridis', origin = 'lower', extent = [0, 357690, 0, 38], aspect=9500)
# ax.set_yticks(np.arange(len(ac_labels)) + 0.5, ac_labels)
# ax.set_yticklabels(ac_labels, va='center', fontsize=4)
# cbar = plt.colorbar(shrink=0.3)
# cbar.ax.tick_params(labelsize=6)
# ax.xaxis.set_tick_params(labelsize=4)
# plt.savefig('Plot_Intersect_PredictedValues_allseqs_dividedbycellcount.png', bbox_inches='tight', dpi = 600)


# seq_nrs = [1060, 1031, 1041, 2149, 1396, 803, 363, 760, 31, 1806] # nr index in array, not original sequence nr
# random_numbers = [random.randint(0, 2514) for _ in range(1)]
# seq_nrs.extend(random_numbers)
# print(seq_nrs)

# def normalize(x):
#     x = np.asarray(x)
#     print(x.shape)
#     return (x - x.min()) / (np.ptp(x))

# for seq_nr in seq_nrs:
#     # seq_nr = 14108      
#     fullname = df_correlations[df_correlations['Index1'] == seq_nr]['Full name'].values[0]
#     original_seq_nr = df_correlations[df_correlations['Index1'] == seq_nr]['Original Seq nr'].values[0]
#     print(fullname, original_seq_nr)

#     pred_values = pred[:, seq_nr]
#     target_values = target[:, seq_nr]
#     maximal_target_value = list(target_values).index(max(target_values))

#     idx_subclass = [51, 60, 61, 62, 63, 64, 43, 45, 46, 44, 48, 49, 50, 47]#     
#     idx_subclass_max = idx_subclass[maximal_target_value]

#     dar_class = df_withnames[df_withnames['Index old'] == idx_subclass_max]['Subclass'].values[0]
#     print(dar_class)

#     corr = np.corrcoef(pred_values, target_values)[0, 1]
#     print(seq_nr, fullname, corr)

#     # pred_values_norm = normalize(pred_values)
#     # target_values_norm = normalize(target_values)

#     df = pd.DataFrame({'Prediction': pred_values.tolist(), 'Target': target_values.tolist()}) # , 'Prediction normalized': pred_values_norm.tolist(), 'Target normalized': target_values_norm.tolist()

#     plt.figure()
#     sns.scatterplot(data = df, x = 'Target', y ='Prediction')
#     plt.title(f'DAR seq {seq_nr}\n Subclass cluster DAR: {fullname} \n Correlation: {corr:.3f} \n Subclass of highest target value: {dar_class}\n Original seq nr: {original_seq_nr}')
#     plt.savefig(f'Plots/DAR_subclass_allenformer_seq{seq_nr}.png', bbox_inches = 'tight')
#     plt.close()

#     # plt.figure()
#     # sns.scatterplot(data = df, x = 'Target normalized', y ='Prediction normalized')
#     # plt.title(f'DAR seq {seq_nr}\n Subclass cluster DAR: {fullname} \n Correlation: {corr:.3f} \n Subclass of highest target value: {dar_class}\n Original seq nr: {original_seq_nr}')
#     # plt.savefig(f'Plots/DAR_subclass_seq{seq_nr}_normalized.png', bbox_inches = 'tight')
#     # plt.close()



# idx_subclass = [35, 29,  30, 36, 37, 39, 40, 33, 25, 38, 41, 42, 31, 27, 28, 32, 8, 9, 10, 5, 6, 11, 12, 13, 19, 17, 15, 16, 21, 18, 20, 3, 1, 2, 52, 56, 57, 59]
# nr_true = 0
# for i, row in enumerate(df_correlations.itertuples()):
#     # print(row)
#     # pred_values = pred[:, i]
#     target_values = target[:, i]
#     maximal_target_value = list(target_values).index(max(target_values))
#     idx_subclass_max = idx_subclass[maximal_target_value]
#     # dar_class = df_ac[df_ac['Index old'] == idx_subclass_max]['names'].values[0]
#     dar_class = df_withnames[df_withnames['Index old'] == idx_subclass_max]['names'].values[0]

#     fullname = df_correlations[df_correlations['Index1'] == i]['Full name'].values[0]
    
#     # print(fullname, dar_class)
#     # print(fullname == dar_class)

#     if fullname == dar_class:
#         nr_true += 1
#     # if i == 100: break
# print(nr_true) 

