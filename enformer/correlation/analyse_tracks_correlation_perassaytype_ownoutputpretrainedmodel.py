import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

## new model == good model

input_file = '/exports/humgen/idenhond/data/Basenji/human-targets.txt'
df = pd.read_csv(input_file, sep = '\t')
df[['assay type', 'description2']] = df.description.str.split(':', n = 1, expand = True) # make new column for assay type

def f(row):
    if row['assay type'] == 'CHIP':
        if any(row['description2'].startswith(x) for x in ['H2AK', 'H2BK', 'H3K', 'H4K']):
            val = 'ChIP Histone'
        else:
            val = 'ChIP TF'
    elif row['assay type'] == 'DNASE' or row['assay type'] == 'ATAC':
        val = 'DNASE/ATAC'
    else:
        val = row['assay type']
    return val

df['assay type split ChIP'] = df.apply(f, axis=1)

print(f'Number of tracks: {df.shape[0]}')
# print(f"Number of trakcs per assay type: \n {df['assay type'].value_counts()}")
print(f"Number of trakcs per assay type split: \n {df['assay type split ChIP'].value_counts()}")

# read csv with correlation score per track for test and validation and train sequences
col_name = ['correlation test']
df_correlation_test = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_own_output_pretrainedmodel.csv', index_col = 0, names = col_name)
col_name = ['correlation validation']
df_correlation_validation = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_own_output_pretrainedmodel.csv', index_col = 0, names = col_name)
col_name = ['correlation train']
df_correlation_train = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train.csv', index_col = 0, names = col_name)
df_correlation_test = df_correlation_test.tail(-1)
df_correlation_validation = df_correlation_validation.tail(-1)
df_correlation_train = df_correlation_train.tail(-1)

# add column to df with test and validation correlation scores
df['test correlation'] = df_correlation_test['correlation test']

df['validation correlation'] = df_correlation_validation['correlation validation']
df['train correlation'] = df_correlation_train['correlation train']
print(df)
# calculate mean test and validation correlation score
# print(f'mean correlation score test: {df["test correlation"].mean(axis=0):.4f}')
# print(f'mean correlation score validation: {df["validation correlation"].mean(axis=0):.4f}')
# print(f'mean correlation score train: {df["train correlation"].mean(axis=0):.4f}')

# print('\nmean correlation score test per assay type: ')
# for key, value in df['assay type split ChIP'].value_counts().to_dict().items():
#     df_subset = df[df['assay type split ChIP'] == key]
#     print(f'mean correlation score test {key}: {df_subset["test correlation"].mean(axis=0):.4f}')

# print('\nmean correlation score valid per assay type: ')
# for key, value in df['assay type split ChIP'].value_counts().to_dict().items():
#     df_subset = df[df['assay type split ChIP'] == key]
#     print(f'mean correlation score valid {key}: {df_subset["validation correlation"].mean(axis=0):.4f}')


# print('\nmean correlation score train per assay type: ')
# for key, value in df['assay type split ChIP'].value_counts().to_dict().items():
#     df_subset = df[df['assay type split ChIP'] == key]
#     print(f'mean correlation score train {key}: {df_subset["train correlation"].mean(axis=0):.4f}')

# read csv with correlation score per track for test and validation and train sequences dnn head
col_name = ['correlation test dnn head']
df_correlation_test_dnn_head = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_dnn_head.csv', index_col = 0, names = col_name)
col_name = ['correlation validation dnn head']
df_correlation_validation_dnn_head = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_dnn_head.csv', index_col = 0, names = col_name)
col_name = ['correlation train dnn head']
df_correlation_train_dnn_head = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train_dnn_head.csv', index_col = 0, names = col_name)

df_correlation_test_dnn_head = df_correlation_test_dnn_head.tail(-1)
df_correlation_validation_dnn_head = df_correlation_validation_dnn_head.tail(-1)
df_correlation_train_dnn_head = df_correlation_train_dnn_head.tail(-1)
# add column to df with test and validation correlation scores

df['test correlation dnn head'] = df_correlation_test_dnn_head['correlation test dnn head']
df['validation correlation dnn head'] = df_correlation_validation_dnn_head['correlation validation dnn head']
df['train correlation dnn head'] = df_correlation_train_dnn_head['correlation train dnn head']

# calculate mean test and validation correlation score
print(f'mean correlation score test dnn head: {df["test correlation dnn head"].mean(axis=0):.4f}')
print(f'mean correlation score validation dnn head: {df["validation correlation dnn head"].mean(axis=0):.4f}')
print(f'mean correlation score train dnn head: {df["train correlation dnn head"].mean(axis=0):.4f}')

print('\nmean correlation score test per assay type dnn head: ')
for key, value in df['assay type split ChIP'].value_counts().to_dict().items():
    df_subset = df[df['assay type split ChIP'] == key]
    print(f'mean correlation score test {key}: {df_subset["test correlation dnn head"].mean(axis=0):.4f}')

print('\nmean correlation score valid per assay type dnn head: ')
for key, value in df['assay type split ChIP'].value_counts().to_dict().items():
    df_subset = df[df['assay type split ChIP'] == key]
    print(f'mean correlation score valid {key}: {df_subset["validation correlation dnn head"].mean(axis=0):.4f}')

print('\nmean correlation score train per assay type train head: ')
for key, value in df['assay type split ChIP'].value_counts().to_dict().items():
    df_subset = df[df['assay type split ChIP'] == key]
    print(f'mean correlation score test {key}: {df_subset["train correlation dnn head"].mean(axis=0):.4f}')