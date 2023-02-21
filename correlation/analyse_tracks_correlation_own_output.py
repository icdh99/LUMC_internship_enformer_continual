import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_file = '/exports/humgen/idenhond/data/Basenji/human-targets.txt'
df = pd.read_csv(input_file, sep = '\t')
df[['assay type', 'description2']] = df.description.str.split(':', n = 1, expand = True) # make new column for assay type

print(f'Number of tracks: {df.shape[0]}')
print(f"Number of trakcs per assay type: \n {df['assay type'].value_counts()}")

# select 10 tracks with highest test correlation score
    # TODO ..... 

# read csv with correlation score per track for test and validation sequences
col_name = ['correlation test']
df_correlation_test = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_own_output.csv', index_col = 0, names = col_name)
col_name = ['correlation validation']
df_correlation_validation = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_own_output.csv', index_col = 0, names = col_name)

df_correlation_test = df_correlation_test.tail(-1)
df_correlation_validation = df_correlation_validation.tail(-1)

# add column to df with test and validation correlation scores
df['test correlation'] = df_correlation_test['correlation test']
df['validation correlation'] = df_correlation_validation['correlation validation']

# calculate mean test and validation correlation score
print(f'mean correlation score test: {df["test correlation"].mean(axis=0):.4f}')
print(f'mean correlation score validation: {df["validation correlation"].mean(axis=0):.4f}')

# plot boxplot with test and boxplot with validation correlation score
    # center line indicates median correlation score 
    # showmeans=True
plt.figure(1)
sns.boxplot(data = df[['test correlation', 'validation correlation']], showmeans = True)
    # boxplot for multiple numerical columns
plt.ylabel('Pearson Correlation Coefficient')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/boxplot_own_output_test_val_corr.png', bbox_inches='tight')

plt.figure(2)
plt.ylabel('Pearson Correlation Coefficient')
sns.violinplot(data = df[['test correlation', 'validation correlation']], orient = 'v')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/violinplot_own_output_test_val_corr.png', bbox_inches='tight')

# plot scatterplot with validation on x and test on y axis (5313 points) for all points and per assay type
plt.figure(3)
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df, x = 'test correlation', y = 'validation correlation')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/scatterplot_own_output_test_val_corr.png', bbox_inches='tight')

plt.figure(4)
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df, x = 'test correlation', y = 'validation correlation', hue = 'assay type')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/scatterplot_own_output_test_val_corr_asssaytype.png', bbox_inches='tight')

for key, value in df['assay type'].value_counts().to_dict().items():
    df_subset = df[df['assay type'] == key]
    plt.figure()
    plt.title(f'Assay type: {key}')
    plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
    sns.scatterplot(data = df_subset, x = 'test correlation', y = 'validation correlation')
    plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/Plots/scatterplot_own_output_test_val_corr_{key}.png', bbox_inches='tight')