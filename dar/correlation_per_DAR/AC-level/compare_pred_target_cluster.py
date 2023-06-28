import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

pred_ac_test = np.loadtxt('/exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/Predictions_test_DAR_aclevel.csv',delimiter=',')
target_ac_test = np.loadtxt('/exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/Targets_test_DAR_aclevel.csv',delimiter=',')
print(pred_ac_test.shape)
print(target_ac_test.shape)

# pred_ac = np.loadtxt('/exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/Predictions_DAR_aclevel.csv',delimiter=',')
# target_ac = np.loadtxt('/exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/Targets_DAR_aclevel.csv',delimiter=',')


# print(pred_ac.shape)
# print(target_ac.shape)

# exit()

idx_subclass = [35, 29,  30, 36, 37, 39, 40, 33, 25, 38, 41, 42, 31, 27, 28, 32, 8, 9, 10, 5, 6, 11, 12, 13, 19, 17, 15, 16, 21, 18, 20, 3, 1, 2, 52, 56, 57, 59] #38

df_withnames = pd.read_csv('/exports/humgen/idenhond/data/basenji_preprocess/human_atac_targets_Ac-level_cluster.csv', sep = '\t')
print(df_withnames)


aantal_goed = 0 
aantal_goed_nr = 0
with open(f'/exports/humgen/idenhond/projects/dar/correlation_per_DAR/AC-level/Clusters_pred_target_ac_test.txt', 'w') as file:
    file.truncate(0)
    header = ['nr', 'target', 'prediction', '\n']
    file.write('\t'.join(header))
    for i in range(15893):
        pred_values = pred_ac_test[:, i]
        target_values = target_ac_test[:, i]

        maximal_target_value = list(target_values).index(max(target_values))
        idx_subclass_max_target = idx_subclass[maximal_target_value]
        maximal_pred_value = list(pred_values).index(max(pred_values))
        idx_subclass_max_pred = idx_subclass[maximal_pred_value]

        class_target = df_withnames[df_withnames['Index old'] == idx_subclass_max_target]['names'].values[0]
        class_pred = df_withnames[df_withnames['Index old'] == idx_subclass_max_pred]['names'].values[0]
        file.write('\t'.join([str(i), class_target, class_pred]))
        file.write('\n')
        if class_pred == class_target: aantal_goed += 1
        if idx_subclass_max_target == idx_subclass_max_pred: aantal_goed_nr += 1
        # if i == 5: break

print(f'aantal goed: {aantal_goed}')
print(f'aantal goed nr: {aantal_goed_nr}')