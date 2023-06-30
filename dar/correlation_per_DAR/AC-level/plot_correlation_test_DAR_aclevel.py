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
class_labels = df_ac['Class'].to_list()

df_counts = pd.read_csv('/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_classes_counts.csv')
df_ac = df_ac.merge(df_counts[['Index old', 'Nuclei']], on = 'Index old')

df_correlations = pd.read_csv('/exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/Correlation_test_DAR_aclevel.csv', index_col = 'Unnamed: 0')
df_correlations['Index1'] = df_correlations.index

pred = np.loadtxt('/exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/Predictions_test_DAR_aclevel.csv',delimiter=',')
target = np.loadtxt('/exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/Targets_test_DAR_aclevel.csv',delimiter=',')
print(pred.shape, target.shape)

fig, ax = plt.subplots()
plt.imshow(pred, interpolation='none', cmap='viridis', origin = 'lower', extent = [0, 15893, 0, 38], aspect=420) # aspect is 15893 / 38, aanpassen voor andere extent values
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
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig4_DAR/Heatmap_DAR_ACLevel_PredictedValues_test.png', bbox_inches='tight', dpi = 800)


fig, ax = plt.subplots()
plt.imshow(target, interpolation='none', cmap='viridis', origin = 'lower', extent = [0, 15893, 0, 38], aspect=420)
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
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig4_DAR/Heatmap_DAR_ACLevel_TargetValues_test.png', bbox_inches='tight', dpi = 800)
plt.savefig('Heatmap_DAR_ACLevel_TargetValues_test.png', bbox_inches='tight', dpi = 800)


# probeersel voor clustermap werkt niet echt nog
# fig, ax = plt.subplots()
# # plt.imshow(pred, interpolation='none', cmap='viridis', origin = 'lower', extent = [0, 357690, 0, 38], aspect=9500)
# sns.clustermap(pred, row_cluster=False)
# ax.set_yticks(np.arange(len(ac_labels)) + 0.5, ac_labels)
# # ax.set_yticklabels([], va='center', fontsize=4)
# # ax.xaxis.set_tick_params(labelsize=3)
# # cbar = plt.colorbar(shrink=0.3, orientation = "horizontal", location = 'top', fraction=0.026, pad=0.06)
# # cbar.outline.set_visible(False)
# # cbar.set_ticks([0, 5])
# # cbar.set_ticklabels([0, 5])
# # cbar.ax.tick_params(size=0, labelsize = 4)
# # cbar.ax.xaxis.set_ticks_position('bottom')
# # cbar.ax.xaxis.set_label_position('bottom')
# # plt.xlabel('DARs', fontsize = 4)
# plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig4_DAR/Heatmap_DAR_ACLevel_PredictedValues_test_clustermap.png', bbox_inches='tight', dpi = 800)

# fig, ax = plt.subplots()
# sns.clustermap(target, row_cluster=False)
# ax.set_yticks(np.arange(len(ac_labels)) + 0.5, ac_labels)
# plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig4_DAR/Heatmap_DAR_ACLevel_TargetValues_test_clustermap.png', bbox_inches='tight', dpi = 800)


exit() 

df_withnames = pd.read_csv('/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_Ac-level_cluster.csv', sep = '\t')
df_correlations['Class'] = df_correlations['Full name'].str.split(' ').str[0]

plt.figure()
ax = sns.histplot(data=df_correlations, x="Correlation", color = 'k', element = 'step')
plt.xlabel('Pearson correlation coefficient', fontsize=9)
plt.ylabel('Count', fontsize=9)
ax.tick_params(axis='both', which='major', labelsize=8)
plt.savefig('Plots/Correlation_test_DAR_aclevel.png', bbox_inches = 'tight', dpi = 300)
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig4_DAR/Correlation_test_DAR_aclevel.png', bbox_inches = 'tight', dpi = 300)
plt.close()

# plt.figure()
# sns.histplot(data=df_correlations, x="Correlation", hue = 'Class')
# plt.savefig('Plots/Correlation_test_DAR_aclevel.png')
# plt.close()

print(df_correlations)
df_correlations['Class'] = df_correlations['Full name'].str.split(' ').str[0]
print(df_correlations)

seq_nr = 14270
fullname = df_correlations[df_correlations['Index1'] == seq_nr]['Full name'].values[0]
original_seq_nr = df_correlations[df_correlations['Index1'] == seq_nr]['Original Seq nr'].values[0]
pred_values = pred[:, seq_nr]
target_values = target[:, seq_nr]
print(pred_values.shape)
print(fullname)
pred_values_sorted = np.sort(pred_values)[::-1]
target_values_sorted = np.sort(target_values)[::-1]
target_values_sorted = target_values_sorted[:5]
for value in target_values_sorted:
    index = list(target_values).index(value)
    idx_subclass_max = idx_subclass[index]
    dar_class = df_withnames[df_withnames['Index old'] == idx_subclass_max]['names'].values[0]
    print(index, idx_subclass_max, dar_class, value)
maximal_target_value = list(target_values).index(max(target_values))
idx_subclass = [35, 29,  30, 36, 37, 39, 40, 33, 25, 38, 41, 42, 31, 27, 28, 32, 8, 9, 10, 5, 6, 11, 12, 13, 19, 17, 15, 16, 21, 18, 20, 3, 1, 2, 52, 56, 57, 59]
idx_subclass_max = idx_subclass[maximal_target_value]
dar_class = df_withnames[df_withnames['Index old'] == idx_subclass_max]['names'].values[0]
corr = np.corrcoef(pred_values, target_values)[0, 1]
df = pd.DataFrame({'Prediction': pred_values.tolist(), 'Target': target_values.tolist()})
plt.figure(figsize=(6,6))
plt.xlabel('Observed')
plt.ylabel('Predicted')
df['class'] = class_labels
ax = sns.scatterplot(data = df, x = 'Target', y ='Prediction', color = 'k', hue = 'class')
ax.text(target_values_sorted[0], pred_values_sorted[0], 'Oligo OPC L1-6 PDGFRA COL20A1   ', fontsize=7, ha = 'right', va = 'center')
ax.text(target_values_sorted[1], pred_values_sorted[2],f'  Oligo Oligo L2-6 OPALIN LOC101927459  ', fontsize=7, ha = 'right', va = 'center')
ax.text(target_values_sorted[2], pred_values_sorted[1], f'Micro-PVM Micro L1-6 TYROBP CD74', fontsize=7, ha = 'right', va = 'center')
plt.text(0.92, 0.01, f'{corr:.3f}', transform=ax.transAxes, fontsize = 9)
# plt.title(f'DAR seq {seq_nr}\n AC-level cluster DAR: {fullname} \n Correlation: {corr:.3f} \n AC-level class of highest target value: {dar_class}\n Original seq nr: {original_seq_nr}')
plt.title(f'DAR cluster: Non-Neuronal, Oligo, Oligo L3-6 SLC1A3 LOC100506725 ')
plt.savefig(f'Plots/DAR_seq{seq_nr}_opgemaakt.png', bbox_inches = 'tight', dpi = 300)
plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig4_DAR/DAR_seq{seq_nr}_opgemaakt.png', bbox_inches = 'tight', dpi = 300)
plt.close()

seq_nr = 7860
fullname = df_correlations[df_correlations['Index1'] == seq_nr]['Full name'].values[0]
original_seq_nr = df_correlations[df_correlations['Index1'] == seq_nr]['Original Seq nr'].values[0]
pred_values = pred[:, seq_nr]
target_values = target[:, seq_nr]
print(pred_values.shape)
print(fullname)
pred_values_sorted = np.sort(pred_values)[::-1]
target_values_sorted = np.sort(target_values)[::-1]
target_values_sorted = target_values_sorted[:5]
for value in target_values_sorted:
    index = list(target_values).index(value)
    idx_subclass_max = idx_subclass[index]
    dar_class = df_withnames[df_withnames['Index old'] == idx_subclass_max]['names'].values[0]
    print(index, idx_subclass_max, dar_class, value)
maximal_target_value = list(target_values).index(max(target_values))
idx_subclass = [35, 29,  30, 36, 37, 39, 40, 33, 25, 38, 41, 42, 31, 27, 28, 32, 8, 9, 10, 5, 6, 11, 12, 13, 19, 17, 15, 16, 21, 18, 20, 3, 1, 2, 52, 56, 57, 59]
idx_subclass_max = idx_subclass[maximal_target_value]
dar_class = df_withnames[df_withnames['Index old'] == idx_subclass_max]['names'].values[0]
corr = np.corrcoef(pred_values, target_values)[0, 1]
df = pd.DataFrame({'Prediction': pred_values.tolist(), 'Target': target_values.tolist()})
plt.figure(figsize=(6,6))
plt.xlabel('Observed')
plt.ylabel('Predicted')
df['class'] = class_labels
ax = sns.scatterplot(data = df, x = 'Target', y ='Prediction', color = 'k', hue = 'class')
ax.text(target_values_sorted[0], pred_values_sorted[0], 'L2/3 IT Exc L2 LINC00507 GLRA3  ', fontsize=7, ha = 'right', va = 'bottom')
ax.text(target_values_sorted[1], pred_values_sorted[3],f'  L6 CT Exc L5-6 FEZF2 \n  C9orf135-AS1', fontsize=7, ha = 'left', va = 'center')
ax.text(target_values_sorted[2], pred_values_sorted[1], f'L2/3 IT Exc L2 LAMP5 KCNG3', fontsize=7, ha = 'left', va = 'top')
plt.text(0.92, 0.01, f'{corr:.3f}', transform=ax.transAxes, fontsize = 9)
# plt.title(f'DAR seq {seq_nr}\n AC-level cluster DAR: {fullname} \n Correlation: {corr:.3f} \n AC-level class of highest target value: {dar_class}\n Original seq nr: {original_seq_nr}')
plt.title(f'DAR cluster: Glutamatergic, L2/3 IT, Exc L2 LINC00507 GLRA3 ')
plt.savefig(f'Plots/DAR_seq{seq_nr}_opgemaakt.png', bbox_inches = 'tight', dpi = 300)
plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig4_DAR/DAR_seq{seq_nr}_opgemaakt.png', bbox_inches = 'tight', dpi = 300)
plt.close()
exit()

df_correlations = df_correlations.sort_values(by = 'Correlation')
print(df_correlations)
print(df_correlations['Correlation'].mean())
print(df_correlations['Correlation'].median())


seq_nrs = [14108, 13385, 11780, 11298, 12626, 14270, 13657] # nr index in array, not original sequence nr
random_numbers = [random.randint(0, 15893) for _ in range(20)]
print(random_numbers)
seq_nrs.extend(random_numbers)
print(seq_nrs)

# for seq_nr in seq_nrs:
#     # seq_nr = 14108      
#     fullname = df_correlations[df_correlations['Index1'] == seq_nr]['Full name'].values[0]
#     original_seq_nr = df_correlations[df_correlations['Index1'] == seq_nr]['Original Seq nr'].values[0]

#     pred_values = pred[:, seq_nr]
#     target_values = target[:, seq_nr]
#     maximal_target_value = list(target_values).index(max(target_values))

#     idx_subclass = [35, 29,  30, 36, 37, 39, 40, 33, 25, 38, 41, 42, 31, 27, 28, 32, 8, 9, 10, 5, 6, 11, 12, 13, 19, 17, 15, 16, 21, 18, 20, 3, 1, 2, 52, 56, 57, 59]
#     idx_subclass_max = idx_subclass[maximal_target_value]

#     dar_class = df_withnames[df_withnames['Index old'] == idx_subclass_max]['names'].values[0]
#     print(dar_class)

#     corr = np.corrcoef(pred_values, target_values)[0, 1]
#     print(seq_nr, fullname, corr)

#     df = pd.DataFrame({'Prediction': pred_values.tolist(), 'Target': target_values.tolist()})

#     plt.figure()
#     sns.scatterplot(data = df, x = 'Target', y ='Prediction')
#     plt.title(f'DAR seq {seq_nr}\n AC-level cluster DAR: {fullname} \n Correlation: {corr:.3f} \n AC-level class of highest target value: {dar_class}\n Original seq nr: {original_seq_nr}')
#     plt.savefig(f'Plots/DAR_seq{seq_nr}.png', bbox_inches = 'tight')
#     plt.close()

# idx_subclass = [35, 29,  30, 36, 37, 39, 40, 33, 25, 38, 41, 42, 31, 27, 28, 32, 8, 9, 10, 5, 6, 11, 12, 13, 19, 17, 15, 16, 21, 18, 20, 3, 1, 2, 52, 56, 57, 59]

# nr_true = 0
# for i, row in enumerate(df_correlations.itertuples()):
#     # print(row)
#     pred_values = pred[:, i]
#     target_values = target[:, i]
#     maximal_target_value = list(pred_values).index(max(pred_values))
#     idx_subclass_max = idx_subclass[maximal_target_value]
#     dar_class = df_withnames[df_withnames['Index old'] == idx_subclass_max]['names'].values[0]

#     fullname = df_correlations[df_correlations['Index1'] == i]['Full name'].values[0]
    
#     # print(fullname, dar_class)
#     # print(fullname == dar_class)

#     if fullname == dar_class:
#         nr_true += 1
#     # if i == 100: break

# print(nr_true)