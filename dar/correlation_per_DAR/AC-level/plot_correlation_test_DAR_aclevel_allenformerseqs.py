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


df_withnames = pd.read_csv('/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_Ac-level_cluster.csv', sep = '\t')
df_correlations = pd.read_csv('/exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/Correlation_DAR_aclevel.csv', index_col = 'Unnamed: 0')
df_correlations['Index1'] = df_correlations.index

# plt.figure()
# ax = sns.histplot(data=df_correlations, x="Correlation", color = 'k', element = 'step')
# plt.xlabel('Pearson correlation coefficient', fontsize=9)
# plt.ylabel('Count', fontsize=9)
# ax.tick_params(axis='both', which='major', labelsize=8)
# plt.savefig('Plots/Correlation_DAR_aclevel.png', bbox_inches = 'tight', dpi = 300)
# plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig4_DAR/Correlation_DAR_aclevel.png', bbox_inches = 'tight', dpi = 300)
# plt.close()

print(f"mean: {df_correlations['Correlation'].mean()}")
print(f"median: {df_correlations['Correlation'].median()}")

pred = np.loadtxt('/exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/Predictions_DAR_aclevel.csv',delimiter=',')
print(pred.shape)  
target = np.loadtxt('/exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/Targets_DAR_aclevel.csv' ,delimiter=',')
print(target.shape)

# fig, ax = plt.subplots()
# plt.imshow(pred, interpolation='none', cmap='viridis', origin = 'lower', extent = [0, 357690, 0, 38], aspect=9500)
# ax.set_yticks(np.arange(len(ac_labels)) + 0.5, ac_labels)
# ax.set_yticklabels([], va='center', fontsize=4)
# ax.xaxis.set_tick_params(labelsize=3)
# cbar = plt.colorbar(shrink=0.3, orientation = "horizontal", location = 'top', fraction=0.026, pad=0.06)
# cbar.outline.set_visible(False)
# cbar.set_ticks([0, 5])
# cbar.set_ticklabels([0, 5])
# cbar.ax.tick_params(size=0, labelsize = 4)
# cbar.ax.xaxis.set_ticks_position('bottom')
# cbar.ax.xaxis.set_label_position('bottom')
# plt.xlabel('DARs', fontsize = 4)
# plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig4_DAR/Heatmap_DAR_ACLevel_PredictedValues_allseqs.png', bbox_inches='tight', dpi = 800)
# plt.savefig('Heatmap_DAR_ACLevel_PredictedValues_allseqs.png', bbox_inches='tight', dpi = 800)

# fig, ax = plt.subplots()
# plt.imshow(target, interpolation='none', cmap='viridis', origin = 'lower', extent = [0, 357690, 0, 38], aspect=9500)
# ax.set_yticks(np.arange(len(ac_labels)) + 0.5, ac_labels)
# ax.set_yticklabels(ac_labels, va='center', fontsize=4)
# ax.xaxis.set_tick_params(labelsize=3)
# cbar = plt.colorbar(shrink=0.3, orientation = "horizontal", location = 'top', fraction=0.026, pad=0.06)
# cbar.outline.set_visible(False)
# cbar.set_ticks([0, 60])
# cbar.set_ticklabels([0, 60])
# cbar.ax.tick_params(size=0, labelsize = 4)
# cbar.ax.xaxis.set_ticks_position('bottom')
# cbar.ax.xaxis.set_label_position('bottom')
# plt.xlabel('DARs', fontsize = 4)
# plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig4_DAR/Heatmap_DAR_ACLevel_TargetValues_allseqs.png', bbox_inches='tight', dpi = 800)
# plt.savefig('Heatmap_DAR_ACLevel_TargetValues_allseqs.png', bbox_inches='tight', dpi = 800)

# df_correlations = df_correlations.sort_values(by = 'Correlation')
print(df_correlations.tail(20))
print(df_correlations[(df_correlations['Correlation'] > 0.8) & (df_correlations['Correlation'] < 0.81)])

seq_nrs = [297991, 282387, 67104, 110812, 277848, 277186, 278286, 292580, 292581, 348897,291401, 277238, 276260, 276259, 276166, 254432, 297773,137650,121419, 161259,  62981  ] # nr index in array, not original sequence nr
random_numbers = [random.randint(0, 357690) for _ in range(1)]
seq_nrs.extend(random_numbers)
print(seq_nrs)

# for seq_nr in seq_nrs:
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
#     plt.savefig(f'Plots/DAR_allseqs_seq{seq_nr}.png', bbox_inches = 'tight')
#     plt.close()

seq_nr = 254432
fullname = df_correlations[df_correlations['Index1'] == seq_nr]['Full name'].values[0]
original_seq_nr = df_correlations[df_correlations['Index1'] == seq_nr]['Original Seq nr'].values[0]
pred_values = pred[:, seq_nr]
target_values = target[:, seq_nr]
pred_values_sorted = np.sort(pred_values)[::-1]
target_values_sorted = np.sort(target_values)[::-1]
target_values_sorted = target_values_sorted[:5]
# print(pred_values.shape)
# print(fullname)
# for value in target_values_sorted:
#     index = list(target_values).index(value)
#     idx_subclass_max = idx_subclass[index]
#     dar_class = df_withnames[df_withnames['Index old'] == idx_subclass_max]['names'].values[0]
#     print(index, idx_subclass_max, dar_class, value)
maximal_target_value = list(target_values).index(max(target_values))
idx_subclass = [35, 29,  30, 36, 37, 39, 40, 33, 25, 38, 41, 42, 31, 27, 28, 32, 8, 9, 10, 5, 6, 11, 12, 13, 19, 17, 15, 16, 21, 18, 20, 3, 1, 2, 52, 56, 57, 59]
idx_subclass_max = idx_subclass[maximal_target_value]
dar_class = df_withnames[df_withnames['Index old'] == idx_subclass_max]['names'].values[0]
corr = np.corrcoef(pred_values, target_values)[0, 1]
df = pd.DataFrame({'Prediction': pred_values.tolist(), 'Target': target_values.tolist()})
df['class'] = class_labels
plt.figure(figsize=(3,3))
plt.xlabel('Observed', fontsize = 8)
plt.ylabel('Predicted', fontsize = 8)
plt.xticks(fontsize = 7)
plt.yticks(fontsize = 7)
ax = sns.scatterplot(data = df, x = 'Target', y ='Prediction', color = 'k', hue = 'class')
ax.text(target_values_sorted[0], pred_values_sorted[0], 'Oligo OPC L1-6 PDGFRA COL20A1  ', fontsize=6, ha = 'right', va = 'center')
ax.text(target_values_sorted[3], pred_values_sorted[1],f'Astro Astro L1-6 FGFR3 PLCG1  ', fontsize=6, ha = 'right', va = 'center')
ax.legend(loc='lower left', bbox_to_anchor=(1, 0), fontsize = 7)
plt.text(0.87, 0.01, f'{corr:.3f}', transform=ax.transAxes, fontsize = 7)
plt.title(f'DAR cluster: \nNon-Neuronal, Oligo, \nOPC L1-6 PDGFRA COL20A1', fontsize = 7)
plt.savefig(f'Plots/DAR_allseqs_seq{seq_nr}_opgemaakt.png', bbox_inches = 'tight', dpi = 300)
plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig4_DAR/DAR_allseqs_seq{seq_nr}_opgemaakt.png', bbox_inches = 'tight', dpi = 300)
plt.close()

seq_nr = 291401
fullname = df_correlations[df_correlations['Index1'] == seq_nr]['Full name'].values[0]
original_seq_nr = df_correlations[df_correlations['Index1'] == seq_nr]['Original Seq nr'].values[0]
pred_values = pred[:, seq_nr]
target_values = target[:, seq_nr]
pred_values_sorted = np.sort(pred_values)[::-1]
target_values_sorted = np.sort(target_values)[::-1]
target_values_sorted = target_values_sorted[:5]
print(pred_values.shape)
print(fullname)
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
df['class'] = class_labels
plt.figure(figsize=(3,3))
plt.xlabel('Observed', fontsize = 8)
plt.ylabel('Predicted', fontsize = 8)
plt.xticks(fontsize = 7)
plt.yticks(fontsize = 7)
ax = sns.scatterplot(data = df, x = 'Target', y ='Prediction', color = 'k', hue = 'class')
ax.text(target_values_sorted[0], pred_values_sorted[0], 'Oligo L2-6 OPALIN LOC101927459  ', fontsize=6, ha = 'right', va = 'center')
ax.text(target_values_sorted[1], pred_values_sorted[1],f'Oligo OPC L1-6 PDGFRA COL20A1  ', fontsize=6, ha = 'right', va = 'center')
plt.text(0.87, 0.01, f'{corr:.3f}', transform=ax.transAxes, fontsize = 7)
plt.title(f'DAR cluster: \nNon-Neuronal Oligo, Oligo, \nL2-6 OPALIN LOC101927459', fontsize = 7)
ax.legend().set_visible(False)
plt.savefig(f'Plots/DAR_allseqs_seq{seq_nr}_opgemaakt.png', bbox_inches = 'tight', dpi = 300)
plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig4_DAR/DAR_allseqs_seq{seq_nr}_opgemaakt.png', bbox_inches = 'tight', dpi = 300)
plt.close()


seq_nr = 121419
fullname = df_correlations[df_correlations['Index1'] == seq_nr]['Full name'].values[0]
original_seq_nr = df_correlations[df_correlations['Index1'] == seq_nr]['Original Seq nr'].values[0]
pred_values = pred[:, seq_nr]
target_values = target[:, seq_nr]
print(pred_values.shape)
print(fullname)
pred_values_sorted = np.sort(pred_values)[::-1]
target_values_sorted = np.sort(target_values)[::-1]
target_values_sorted = target_values_sorted[:5]
print(target_values_sorted)
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
plt.figure(figsize=(3,3))
plt.xlabel('Observed', fontsize = 8)
plt.ylabel('Predicted', fontsize = 8)
plt.xticks(fontsize = 7)
plt.yticks(fontsize = 7)
df['class'] = class_labels
ax = sns.scatterplot(data = df, x = 'Target', y ='Prediction', color = 'k', hue = 'class')
ax.text(target_values_sorted[0], pred_values_sorted[1], 'L2/3 IT Exc L2 LAMP5 KCNG3  ', fontsize=6, ha = 'right', va = 'center')
ax.text(target_values_sorted[1], pred_values_sorted[0],f'  L2/3 IT Exc L2 LINC00507 GLRA3  ', fontsize=6, ha = 'right', va = 'center')
# ax.text(target_values_sorted[2], pred_values_sorted[2]-0.015, f'L2/3 IT Exc L2-3 LINC00507 DSG3', fontsize=7, ha = 'right', va = 'top')
ax.legend().set_visible(False)
plt.text(0.87, 0.01, f'{corr:.3f}', transform=ax.transAxes, fontsize = 7)
plt.title(f'DAR cluster: \nGlutamatergic, L2/3 IT, \nExc L2 LAMP5 KCNG3', fontsize = 7)
plt.savefig(f'Plots/DAR_allseqs_seq{seq_nr}_opgemaakt.png', bbox_inches = 'tight', dpi = 300)
plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig4_DAR/DAR_allseqs_seq{seq_nr}_opgemaakt.png', bbox_inches = 'tight', dpi = 300)
plt.close()



seq_nr = 297773
fullname = df_correlations[df_correlations['Index1'] == seq_nr]['Full name'].values[0]
original_seq_nr = df_correlations[df_correlations['Index1'] == seq_nr]['Original Seq nr'].values[0]
pred_values = pred[:, seq_nr] # (38,)
target_values = target[:, seq_nr]
print(pred_values.shape)
print(fullname)
pred_values_sorted = np.sort(pred_values)[::-1]
target_values_sorted = np.sort(target_values)
target_values_sorted = target_values_sorted[::-1]
target_values_sorted = target_values_sorted[:5]
for value in target_values_sorted:
    index = list(target_values).index(value)
    idx_subclass_max = idx_subclass[index]
    dar_class = df_withnames[df_withnames['Index old'] == idx_subclass_max]['names'].values[0]
    print(index, idx_subclass_max, dar_class)
maximal_target_value = list(target_values).index(max(target_values))
idx_subclass = [35, 29,  30, 36, 37, 39, 40, 33, 25, 38, 41, 42, 31, 27, 28, 32, 8, 9, 10, 5, 6, 11, 12, 13, 19, 17, 15, 16, 21, 18, 20, 3, 1, 2, 52, 56, 57, 59]
idx_subclass_max = idx_subclass[maximal_target_value]
dar_class = df_withnames[df_withnames['Index old'] == idx_subclass_max]['names'].values[0]
corr = np.corrcoef(pred_values, target_values)[0, 1]
df = pd.DataFrame({'Prediction': pred_values.tolist(), 'Target': target_values.tolist()})
plt.figure(figsize=(3,3))
plt.xlabel('Observed', fontsize = 8)
plt.ylabel('Predicted', fontsize = 8)
plt.xticks(fontsize = 7)
plt.yticks(fontsize = 7)
df['class'] = class_labels
ax = sns.scatterplot(data = df, x = 'Target', y ='Prediction', color = 'k', hue = 'class')
ax.text(target_values_sorted[0], pred_values_sorted[0],'OPC L1-6 PDGFRA COL20A1  ', fontsize=6, ha = 'right', va = 'center')
ax.text(target_values_sorted[1], pred_values_sorted[4],f'\nOligo L2-6\nOPALIN\nLOC101927459  ', fontsize=6, ha = 'left', va = 'top')
plt.text(0.87, 0.01, f'{corr:.3f}', transform=ax.transAxes, fontsize = 7)
# plt.title(f'DAR seq {seq_nr}\n AC-level cluster DAR: {fullname} \n Correlation: {corr:.3f} \n AC-level class of highest target value: {dar_class}\n Original seq nr: {original_seq_nr}')
plt.title(f'DAR cluster: \nNon-Neuronal Oligo, Oligo, \nL2-6 OPALIN LOC101927459', fontsize = 7)
ax.legend().set_visible(False)
plt.savefig(f'Plots/DAR_allseqs_seq{seq_nr}_opgemaakt.png', bbox_inches = 'tight', dpi = 300)
plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig4_DAR/DAR_allseqs_seq{seq_nr}_opgemaakt.png', bbox_inches = 'tight', dpi = 300)
plt.close()

seq_nr = 31
fullname = df_correlations[df_correlations['Index1'] == seq_nr]['Full name'].values[0]
original_seq_nr = df_correlations[df_correlations['Index1'] == seq_nr]['Original Seq nr'].values[0]
pred_values = pred[:, seq_nr] # (38,)
target_values = target[:, seq_nr]
print(pred_values.shape)
print(fullname)
pred_values_sorted = np.sort(pred_values)[::-1]
target_values_sorted = np.sort(target_values)
target_values_sorted = target_values_sorted[::-1]
target_values_sorted = target_values_sorted[:5]
for value in target_values_sorted:
    index = list(target_values).index(value)
    idx_subclass_max = idx_subclass[index]
    dar_class = df_withnames[df_withnames['Index old'] == idx_subclass_max]['names'].values[0]
    print(index, idx_subclass_max, dar_class)
maximal_target_value = list(target_values).index(max(target_values))
idx_subclass = [35, 29,  30, 36, 37, 39, 40, 33, 25, 38, 41, 42, 31, 27, 28, 32, 8, 9, 10, 5, 6, 11, 12, 13, 19, 17, 15, 16, 21, 18, 20, 3, 1, 2, 52, 56, 57, 59]
idx_subclass_max = idx_subclass[maximal_target_value]
dar_class = df_withnames[df_withnames['Index old'] == idx_subclass_max]['names'].values[0]
corr = np.corrcoef(pred_values, target_values)[0, 1]
df = pd.DataFrame({'Prediction': pred_values.tolist(), 'Target': target_values.tolist()})
plt.figure(figsize=(3,3))
plt.xlabel('Observed', fontsize = 8)
plt.ylabel('Predicted', fontsize = 8)
plt.xticks(fontsize = 7)
plt.yticks(fontsize = 7)
df['class'] = class_labels
ax = sns.scatterplot(data = df, x = 'Target', y ='Prediction', color = 'k', hue = 'class')
ax.text(target_values_sorted[0], pred_values_sorted[0],'LAMP5 Inh L1-6 LAMP5 KIRREL  ', fontsize=6, ha = 'right', va = 'center')
# ax.text(target_values_sorted[1], pred_values_sorted[4],f'  Oligo L2-6\n  OPALIN\n  LOC101927459  ', fontsize=6, ha = 'left', va = 'top')
plt.text(0.87, 0.01, f'{corr:.3f}', transform=ax.transAxes, fontsize = 7)
# plt.title(f'DAR seq {seq_nr}\n AC-level cluster DAR: {fullname} \n Correlation: {corr:.3f} \n AC-level class of highest target value: {dar_class}\n Original seq nr: {original_seq_nr}')
plt.title(f'DAR cluster: \nGABAergic, LAMP5, \nInh L1-6 LAMP5 KIRREL', fontsize = 7)
ax.legend().set_visible(False)
plt.savefig(f'Plots/DAR_allseqs_seq{seq_nr}_opgemaakt.png', bbox_inches = 'tight', dpi = 300)
plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig4_DAR/DAR_allseqs_seq{seq_nr}_opgemaakt.png', bbox_inches = 'tight', dpi = 300)
plt.close()

# seq_nr = 297991
# fullname = df_correlations[df_correlations['Index1'] == seq_nr]['Full name'].values[0]
# original_seq_nr = df_correlations[df_correlations['Index1'] == seq_nr]['Original Seq nr'].values[0]
# pred_values = pred[:, seq_nr] # (38,)
# target_values = target[:, seq_nr]
# print(pred_values.shape)
# print(fullname)
# pred_values_sorted = np.sort(pred_values)[::-1]
# target_values_sorted = np.sort(target_values)
# target_values_sorted = target_values_sorted[::-1]
# target_values_sorted = target_values_sorted[:5]
# for value in target_values_sorted:
#     index = list(target_values).index(value)
#     idx_subclass_max = idx_subclass[index]
#     dar_class = df_withnames[df_withnames['Index old'] == idx_subclass_max]['names'].values[0]
#     print(index, idx_subclass_max, dar_class)
# maximal_target_value = list(target_values).index(max(target_values))
# idx_subclass = [35, 29,  30, 36, 37, 39, 40, 33, 25, 38, 41, 42, 31, 27, 28, 32, 8, 9, 10, 5, 6, 11, 12, 13, 19, 17, 15, 16, 21, 18, 20, 3, 1, 2, 52, 56, 57, 59]
# idx_subclass_max = idx_subclass[maximal_target_value]
# dar_class = df_withnames[df_withnames['Index old'] == idx_subclass_max]['names'].values[0]
# corr = np.corrcoef(pred_values, target_values)[0, 1]
# df = pd.DataFrame({'Prediction': pred_values.tolist(), 'Target': target_values.tolist()})
# plt.figure(figsize=(3,3))
# plt.xlabel('Observed', fontsize = 6)
# plt.ylabel('Predicted', fontsize = 6)
# plt.xticks(fontsize = 6)
# plt.yticks(fontsize = 6)
# df['class'] = class_labels
# ax = sns.scatterplot(data = df, x = 'Target', y ='Prediction', color = 'k', hue = 'class')
# plt.text(0.85, 0.01, f'{corr:.3f}', transform=ax.transAxes, fontsize = 7)
# plt.title(f'DAR cluster: \nNon-Neuronal, Oligo,\nOligo L2-6 OPALIN LOC101927459', fontsize = 7)
# ax.legend().set_visible(False)
# plt.savefig(f'Plots/DAR_allseqs_seq{seq_nr}_opgemaakt.png', bbox_inches = 'tight', dpi = 300)
# plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig4_DAR/DAR_allseqs_seq{seq_nr}_opgemaakt.png', bbox_inches = 'tight', dpi = 300)
# plt.close()

seq_nr = 1041
fullname = df_correlations[df_correlations['Index1'] == seq_nr]['Full name'].values[0]
original_seq_nr = df_correlations[df_correlations['Index1'] == seq_nr]['Original Seq nr'].values[0]
pred_values = pred[:, seq_nr] # (38,)
target_values = target[:, seq_nr]
print(pred_values.shape)
print(fullname)
pred_values_sorted = np.sort(pred_values)[::-1]
target_values_sorted = np.sort(target_values)
target_values_sorted = target_values_sorted[::-1]
target_values_sorted = target_values_sorted[:5]
for value in target_values_sorted:
    index = list(target_values).index(value)
    idx_subclass_max = idx_subclass[index]
    dar_class = df_withnames[df_withnames['Index old'] == idx_subclass_max]['names'].values[0]
    print(index, idx_subclass_max, dar_class)
maximal_target_value = list(target_values).index(max(target_values))
idx_subclass = [35, 29,  30, 36, 37, 39, 40, 33, 25, 38, 41, 42, 31, 27, 28, 32, 8, 9, 10, 5, 6, 11, 12, 13, 19, 17, 15, 16, 21, 18, 20, 3, 1, 2, 52, 56, 57, 59]
idx_subclass_max = idx_subclass[maximal_target_value]
dar_class = df_withnames[df_withnames['Index old'] == idx_subclass_max]['names'].values[0]
corr = np.corrcoef(pred_values, target_values)[0, 1]
df = pd.DataFrame({'Prediction': pred_values.tolist(), 'Target': target_values.tolist()})
plt.figure(figsize=(3,3))
plt.xlabel('Observed', fontsize = 6)
plt.ylabel('Predicted', fontsize = 6)
plt.xticks(fontsize = 6)
plt.yticks(fontsize = 6)
df['class'] = class_labels
ax = sns.scatterplot(data = df, x = 'Target', y ='Prediction', color = 'k', hue = 'class')
ax.text(target_values_sorted[0], pred_values_sorted[8],'LAMP5 Inh L1-6 LAMP5 KIRREL  ', fontsize=6, ha = 'right', va = 'center')
plt.text(0.87, 0.01, f'{corr:.3f}', transform=ax.transAxes, fontsize = 7)
plt.title(f'DAR cluster: \nGABAergic, LAMP5, \nInh L1-6 LAMP5 KIRREL', fontsize = 7)
ax.legend().set_visible(False)
plt.savefig(f'Plots/DAR_allseqs_seq{seq_nr}_opgemaakt.png', bbox_inches = 'tight', dpi = 300)
plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig4_DAR/DAR_allseqs_seq{seq_nr}_opgemaakt.png', bbox_inches = 'tight', dpi = 300)
plt.close()