import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# df_newtrack0 = pd.read_csv('matrix_newtrack0.csv', header = None, names = ['value'])

# print(df_newtrack0)


# sns.boxplot(data = df_newtrack0, x = 'value'  )
# plt.xlim(-0.1, 0.1)
# plt.savefig('test.png')


filenames = ['matrix_newtrack0.csv', 'matrix_newtrack1.csv', 'matrix_newtrack2.csv', 
             'matrix_newtrack3.csv', 'matrix_newtrack4.csv', 'matrix_newtrack5.csv', 
             'matrix_newtrack6.csv', 'matrix_newtrack6.csv', 'matrix_newtrack8.csv', 
             'matrix_newtrack9.csv', 'matrix_newtrack10.csv', 'matrix_newtrack11.csv', 
             'matrix_newtrack12.csv', 'matrix_newtrack13.csv', 'matrix_newtrack14.csv', 
             'matrix_newtrack15.csv', 'matrix_newtrack16.csv', 'matrix_newtrack17.csv', 
             'matrix_newtrack18.csv', ]


num_files = len(filenames)
num_rows = (num_files + 3) // 4  # Round up to the nearest integer
num_cols = 4

# Create the subplot grid
fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8), sharex=True, sharey=True)

# Flatten the axes array for easier indexing
axes = axes.ravel()

PROPS = {
    'boxprops':{'facecolor':'none', 'edgecolor':'black'},
    'medianprops':{'color':'black'},
    'whiskerprops':{'color':'black'},
    'capprops':{'color':'black'}
}

df_targets = pd.read_csv('/exports/humgen/idenhond/data/Basenji/human-targets.txt', sep = '\t', index_col = 'index')
df_targets[['assay type', 'description2']] = df_targets.description.str.split(':', n = 1, expand = True) # make new column for assay type
def f(row):
    if row['assay type'] == 'CHIP':
        if any(row['description2'].startswith(x) for x in ['H2AK', 'H2BK', 'H3K', 'H4K']): val = 'ChIP Histone'
        else: val = 'ChIP TF'
    elif row['assay type'] == 'DNASE' or row['assay type'] == 'ATAC': val = 'DNASE_ATAC'
    else: val = row['assay type']
    return val
df_targets['assay type split ChIP'] = df_targets.apply(f, axis=1)
print(df_targets)

for i, file_name in enumerate(filenames):
    if i < num_files:
        df = pd.read_csv(file_name, header = None, names = ['value'])
        
        df['assay type'] = df_targets['assay type split ChIP']
        print(df)
        data = df['value']
        
        
        # sns.boxplot(data = df, x='value', hue = 'assay type', orient='h', ax=axes[i], color = 'k',  **PROPS)
        sns.boxplot(data = df, x='value',  orient='h', y = 'assay type', ax=axes[i], width = 0.8)
        
        axes[i].set_title(f'Track {i+1}')
        
        axes[i].set_xlabel('')
        
        # Set the x-axis label for the bottom row of plots and for track 16
        if i >= (num_rows - 1) * num_cols:
            axes[i].set_xlabel('Correlation')
            axes[i].tick_params(axis='x', rotation=45) 

        if i == 15:
            axes[i].set_xlabel('Correlation')
            axes[i].tick_params(axis='x', rotation=45) 

        # axes[i].legend_.remove()

        

fig.suptitle('Pearson correlation coefficient between each new track and 5,313 old tracks')

fig.delaxes(axes[-1])

plt.tight_layout()
plt.xlim(-0.01, 0.01)
# plt.legend()
plt.savefig('boxplot_correlationnewvsol_assaytype.png')