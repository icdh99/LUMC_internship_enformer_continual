import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

random.seed(18)

df_correlations = pd.read_csv('/exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/Correlation_test_DAR_aclevel.csv', index_col = 'Unnamed: 0')
df_correlations['Index1'] = df_correlations.index
print(df_correlations)

pred = np.loadtxt('/exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/Predictions_test_DAR_aclevel.csv',delimiter=',')
print(pred.shape)  

target = np.loadtxt('/exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/Targets_test_DAR_aclevel.csv',delimiter=',')
print(target.shape)

df_withnames = pd.read_csv('/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_Ac-level_cluster.csv', sep = '\t')
print(df_withnames)

df_correlations['Class'] = df_correlations['Full name'].str.split(' ').str[0]
print(df_correlations)

plt.figure()
sns.histplot(data=df_correlations, x="Correlation", hue = 'Class')
plt.savefig('Plots/Correlation_test_DAR_aclevel.png')
plt.close()

print(df_correlations)
# df_correlations[['Class']] = df_correlations['Full name'].str.split(' ', 1)
df_correlations['Class'] = df_correlations['Full name'].str.split(' ').str[0]
print(df_correlations)
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