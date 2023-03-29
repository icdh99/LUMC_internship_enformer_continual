import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

input_file = '/exports/humgen/idenhond/data/basenji_preprocess/output_tfr/targets.txt'
df = pd.read_csv(input_file, sep = '\t')
df = df.drop([2,3,4]).reset_index()
t = 'ChIP TF'
h = 'ChIP Histone'
d = 'DNASE'
assaytype = [h,h,t,h,t,h,t,t,t,h,h,h,h,h,t,t,d,d,d]
df['assay type'] = assaytype

print(f'Number of tracks: {df.shape[0]}')
print(f"Number of trakcs per assay type: \n {df['assay type'].value_counts()}\n")

df_correlation_test = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_dnn_head_newtracks.csv', index_col = 0, names = ['correlation test']).tail(-1)
df_correlation_valid = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_valid_dnn_head_newtracks.csv', index_col = 0, names = ['correlation valid']).tail(-1)
df_correlation_train = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_train_dnn_head_newtracks.csv', index_col = 0, names = ['correlation train']).tail(-1)

df['test correlation'] = df_correlation_test['correlation test']
df['valid correlation'] = df_correlation_valid['correlation valid']
df['train correlation'] = df_correlation_train['correlation train']

print(df)

plt.figure(1)
sns.boxplot(data = df[['test correlation', 'valid correlation', 'train correlation']], showmeans = True)
    # boxplot for multiple numerical columns
plt.ylabel('Pearson Correlation Coefficient')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_newtracks/dnn_newtracks_boxplot.png', bbox_inches='tight')
plt.close

plt.figure(2)
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df, x = 'test correlation', y = 'valid correlation')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_newtracks/dnn_newtracks_scatterplot_testval.png', bbox_inches='tight')
plt.close()

plt.figure(3)
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df, x = 'test correlation', y = 'valid correlation', hue = 'assay type')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_newtracks/dnn_newtracks_scatterplot_assaytype_testval.png', bbox_inches='tight')
plt.close()

plt.figure(4)
plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
sns.scatterplot(data = df, x = 'test correlation', y = 'train correlation', hue = 'assay type')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_newtracks/dnn_newtracks_scatterplot_assaytype_testtrain.png', bbox_inches='tight')
plt.close()

df_long = pd.melt(df, id_vars = ['index', 'assay type'], value_vars = ['train correlation', 'test correlation', 'valid correlation'], var_name = 'subset', value_name = 'correlation')
print(df_long)

plt.figure(5)
sns.catplot(data = df_long, x = 'subset', y = 'correlation', hue = 'assay type')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_newtracks/dnn_newtracks_catplot.png', bbox_inches='tight')
plt.close()

plt.figure(6)
sns.catplot(data = df_long, x = 'subset', y = 'correlation', hue = 'index')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_newtracks/dnn_newtracks_catplot_index.png', bbox_inches='tight')
plt.close()

plt.figure(7)
sns.catplot(data = df_long, x = 'subset', y = 'correlation',  kind = 'point')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_newtracks/dnn_newtracks_catlineplot.png', bbox_inches='tight')
plt.close()

plt.figure(8)
sns.lineplot(data = df_long, x = 'subset', y = 'correlation', hue = 'index')
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_newtracks/dnn_newtracks_lineplot.png', bbox_inches='tight')


