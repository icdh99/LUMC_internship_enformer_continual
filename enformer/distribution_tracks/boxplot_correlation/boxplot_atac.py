import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

np.set_printoptions(linewidth=400) 

# filename_targets = '/exports/humgen/idenhond/projects/enformer/distribution_tracks/boxplot_correlation/final_correlation_matrix_testset_atac_targets.csv'
# targets = np.loadtxt(filename_targets, delimiter=',')
# filename_outputs = '/exports/humgen/idenhond/projects/enformer/distribution_tracks/boxplot_correlation/final_correlation_matrix_testset_atac_outputs.csv'
# outputs = np.loadtxt(filename_outputs, delimiter=',')

# print(targets.shape)
# print(outputs.shape)

# values_targets = targets[np.tril_indices(66, k=-1)].tolist()

# values_outputs = outputs[np.tril_indices(66, k=-1)].tolist()

# data = {"List": ['ATAC'] * len(values_outputs) * 2 ,
#         "Value": values_targets + values_outputs,
#         "Type": ['TARGET'] * len(values_outputs) + ['OUTPUT'] * len(values_outputs) }

# df = pd.DataFrame(data)
# plt.figure()
# flierprops = dict(marker='o', markerfacecolor='None', markersize=0.5,  markeredgecolor='black')
# ax = sns.boxplot(data = df, x = 'List', y = 'Value', hue = 'Type', palette = sns.color_palette("Paired"), flierprops=flierprops)
# # plt.xticks(rotation = 45)
# plt.title('Correlation between true and predicted genomic tracks per assay type')
# plt.xlabel(None)
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles[0:], labels=labels[0:])
# plt.ylabel('Correlation')
# plt.savefig('boxplot_atac.png', bbox_inches = 'tight')
# plt.close()

# plot for different levels 
df = pd.read_csv('/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_classes_counts.csv', sep = ',')
df.drop_duplicates(subset=['Index old'], keep = 'first', inplace=True)
print(df)
print(df['level'].value_counts())

filename_targets = '/exports/humgen/idenhond/projects/enformer/distribution_tracks/boxplot_correlation/final_correlation_matrix_testset_atac_targets_ordered.csv'
targets = np.loadtxt(filename_targets, delimiter=',')
filename_outputs = '/exports/humgen/idenhond/projects/enformer/distribution_tracks/boxplot_correlation/final_correlation_matrix_testset_atac_outputs_ordered.csv'
outputs = np.loadtxt(filename_outputs, delimiter=',')

print(targets.shape)
print(outputs.shape)

targets_class = targets[:3, :3]
targets_subclass = targets[3:23, 3:23]
targets_aclevel = targets[23:66, 23:66]
targets_class_vs_subclass = targets[:3, 3:23]
targets_class_vs_aclevel = targets[:3, 23:66]
targets_subclass_vs_aclevel = targets[3:23, 23:66]

outputs_class = outputs[:3, :3]
outputs_subclass = outputs[3:23, 3:23]
outputs_aclevel = outputs[23:66, 23:66]
outputs_class_vs_subclass = outputs[:3, 3:23]
outputs_class_vs_aclevel = outputs[:3, 23:66]
outputs_subclass_vs_aclevel = outputs[3:23, 23:66]

values_targets = targets[np.tril_indices(66, k=-1)].tolist()
values_targets_class = targets_class[np.tril_indices(3, k=-1)].tolist()
values_targets_subclass = targets_subclass[np.tril_indices(20, k=-1)].tolist()
values_targets_aclevel = targets_aclevel[np.tril_indices(43, k=-1)].tolist()
values_outputs = outputs[np.tril_indices(66, k=-1)].tolist()
values_outputs_class = outputs_class[np.tril_indices(3, k=-1)].tolist()
values_outputs_subclass = outputs_subclass[np.tril_indices(20, k=-1)].tolist()
values_outputs_aclevel = outputs_aclevel[np.tril_indices(43, k=-1)].tolist()


data = {"List": ['All'] * len(values_outputs) * 2 + ['Class'] * len(values_targets_class) * 2 + ['Subclass'] * len(values_targets_subclass) * 2 + ['AC level'] * len(values_targets_aclevel) * 2 ,
        "Value": values_targets + values_outputs + values_targets_class + values_outputs_class + values_targets_subclass + values_outputs_subclass + values_targets_aclevel + values_outputs_aclevel,
        "Type": ['Observed'] * len(values_outputs) + ['Predicted'] * len(values_outputs) + ['Observed'] * len(values_outputs_class) + ['Predicted'] * len(values_outputs_class) + ['Observed'] * len(values_outputs_subclass) + ['Predicted'] * len(values_outputs_subclass) + ['Observed'] * len(values_outputs_aclevel) + ['Predicted'] * len(values_outputs_aclevel) }

df = pd.DataFrame(data)
fig = plt.figure(figsize = (10.0, 5.8))
plt.tight_layout()
flierprops = dict(marker='o', markerfacecolor='None', markersize=0.5,  markeredgecolor='black')
ax = sns.boxplot(data = df, x = 'List', y = 'Value', hue = 'Type', palette = sns.color_palette("Paired"), flierprops=flierprops)
# plt.xticks(rotation = 45)
plt.xlabel(None)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[0:], labels=labels[0:])
plt.legend(loc = 'lower left', bbox_to_anchor=(0, 1, 0.5, 0.5))
plt.ylabel('Correlation')
plt.savefig('boxplot_atac_ordered.png', bbox_inches = 'tight', dpi = 300)
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig3_ATAC/boxplot_atac_ordered.png', bbox_inches = 'tight', dpi = 300)
plt.close()
