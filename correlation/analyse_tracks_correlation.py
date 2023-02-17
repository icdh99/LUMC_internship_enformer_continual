import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_file = '/exports/humgen/idenhond/data/Basenji/human-targets.txt'
df = pd.read_csv(input_file, sep = '\t')
df[['assay type', 'description2']] = df.description.str.split(':', n = 1, expand = True) # make new column assay type

print(f'Number of tracks: {df.shape[0]}')
print(f"Number of trakcs per assay type: \n {df['assay type'].value_counts()}")

exit()
# VANAF HIER ALLEEN OPZET

# select 10 tracks with highest test correlation score


# read csv with correlation score per track for test and validation sequences
df_correlation_test = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation')
df_correlation_validation = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation')

# add column to df with test and validation correlation scores
df['test correlation'] = df_correlation_test
df['validation correlation'] = df_correlation_validation
print(df)

# calculate mean test and validation correlation score
print(f'mean correlation score test: {df["test correlation"].mean(axis=0)}')
print(f'mean correlation score validation: {df["validation correlation"].mean(axis=0)}')

# plot boxplot with test and boxplot with validation correlation score
    # center line indicates median correlation score 
    # showmeans=True
plt.figure(1)
sns.boxplot(data = df[['test correlation', 'validation correlation']], showmeans = True)
    # boxplot for multiple numerical columns
    # test orient = 'v' as argument
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/boxplot_test_val_corr.png', bbox_inches='tight')

# plot scatterplot with validation on x and test on y axis (5313 points)
plt.figure(2)
sns.scatterplot(data = df, x = 'test correlation', y = 'validation correlation')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/scatterplot_test_val_corr.png', bbox_inches='tight')
plt.figure(3)
sns.scatterplot(data = df, x = 'test correlation', y = 'validation correlation', hue = 'assay type')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/scatterplot_test_val_corr_asssaytype.png', bbox_inches='tight')
