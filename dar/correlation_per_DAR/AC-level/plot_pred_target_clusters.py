from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


# labels
df_labels = pd.read_csv('/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_Ac-level_cluster_noduplicates.csv', sep = '\t')
labels_list = df_labels['names'].tolist()

# enformer test sequences
df = pd.read_csv('/exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/Clusters_pred_target_ac_test.txt', sep = '\t')
preds = df['prediction'].tolist()
targets = df['target'].tolist()
print(df['target'].value_counts())
print(df['prediction'].value_counts())
cm = confusion_matrix(targets, preds, labels = labels_list)
cm_df = pd.DataFrame(cm, index = labels_list, columns = labels_list)
fig, ax = plt.subplots(figsize = (10,10))
cpalette = sns.color_palette("Blues", as_cmap=True)
ax = sns.heatmap(cm_df, annot = False, square = True, cmap = cpalette, cbar_kws={"shrink": 0.7})
ax.tick_params(axis='both', which='major', labelsize=7)
plt.xlabel('AC-level cluster of highest predicted value', fontsize = 8)
plt.ylabel('AC-level cluster of highest observed value', fontsize = 8)
plt.savefig('cm_test_ac.png', bbox_inches = 'tight', dpi = 300)
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig4_DAR/cm_test_ac.png', bbox_inches = 'tight', dpi = 300)
plt.close()

# all enformer sequences
df = pd.read_csv('/exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/Clusters_pred_target_ac.txt', sep = '\t')
preds = df['prediction'].tolist()
targets = df['target'].tolist()
cm = confusion_matrix(targets, preds)
cm_df = pd.DataFrame(cm, index = labels_list, columns = labels_list)
fig, ax = plt.subplots(figsize = (10,10))
cpalette = sns.color_palette("Blues", as_cmap=True)
ax = sns.heatmap(cm_df, annot = False, square = True, cmap = cpalette, cbar_kws={"shrink": 0.7})
ax.tick_params(axis='both', which='major', labelsize=7)
plt.xlabel('AC-level cluster of highest predicted value', fontsize = 8)
plt.ylabel('AC-level cluster of highest observed value', fontsize = 8)
plt.savefig('cm_ac.png', bbox_inches = 'tight', dpi = 300)
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig4_DAR/cm_ac.png', bbox_inches = 'tight', dpi = 300)
plt.close()