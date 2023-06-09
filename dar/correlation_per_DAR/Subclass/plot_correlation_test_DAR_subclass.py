import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

random.seed(18)


df_correlations = pd.read_csv('/exports/humgen/idenhond/projects/dar/correlation_per_DAR/Subclass/Correlation_test_DAR_subclass.csv', index_col = 'Unnamed: 0')
df_correlations['Index1'] = df_correlations.index
print(df_correlations)

pred = np.loadtxt('/exports/humgen/idenhond/projects/dar/correlation_per_DAR/Subclass/Predictions_test_DAR_subclass.csv',delimiter=',')
print(pred.shape)  

target = np.loadtxt('/exports/humgen/idenhond/projects/dar/correlation_per_DAR/Subclass/Targets_test_DAR_subclass.csv' ,delimiter=',')
print(target.shape)

df_withnames = pd.read_csv('/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_Subclass.csv', sep = '\t')
print(df_withnames)

plt.figure()
sns.histplot(data=df_correlations, x="Correlation")
plt.savefig('Plots/Correlation_test_DAR_subclass.png')
plt.close()

df_correlations = df_correlations.sort_values(by = 'Correlation')
print(df_correlations)
print(f"mean: {df_correlations['Correlation'].mean()}")
print(f"median: {df_correlations['Correlation'].median()}")





seq_nrs = [1060, 1031, 1041, 2149, 1396, 803, 363, 760, 31, 1806] # nr index in array, not original sequence nr
random_numbers = [random.randint(0, 2514) for _ in range(1)]
print(random_numbers)
seq_nrs.extend(random_numbers)
print(seq_nrs)

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
#     plt.savefig(f'Plots/DAR_subclass_seq{seq_nr}.png', bbox_inches = 'tight')
#     plt.close()

    # plt.figure()
    # sns.scatterplot(data = df, x = 'Target normalized', y ='Prediction normalized')
    # plt.title(f'DAR seq {seq_nr}\n Subclass cluster DAR: {fullname} \n Correlation: {corr:.3f} \n Subclass of highest target value: {dar_class}\n Original seq nr: {original_seq_nr}')
    # plt.savefig(f'Plots/DAR_subclass_seq{seq_nr}_normalized.png', bbox_inches = 'tight')
    # plt.close()



idx_subclass = [51, 60, 61, 62, 63, 64, 43, 45, 46, 44, 48, 49, 50, 47]
nr_true = 0
for i, row in enumerate(df_correlations.itertuples()):
    # print(row)
    pred_values = pred[:, i]
    target_values = target[:, i]
    maximal_target_value = list(target_values).index(max(target_values))
    idx_subclass_max = idx_subclass[maximal_target_value]
    dar_class = df_withnames[df_withnames['Index old'] == idx_subclass_max]['Subclass'].values[0]

    fullname = df_correlations[df_correlations['Index1'] == i]['Full name'].values[0]
    
    # print(fullname, dar_class)
    # print(fullname == dar_class)

    if fullname == dar_class:
        nr_true += 1
    # if i == 100: break
print(nr_true) #1081

