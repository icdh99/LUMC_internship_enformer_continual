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
df_correlation_test_alltracksmodel = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_dnn_head_alltracks.csv', index_col = 0, names = ['correlation test all tracks model']).tail(-1)
df_correlation_test_alltracksmodel = df_correlation_test_alltracksmodel.tail(19).reset_index()

df['test correlation'] = df_correlation_test['correlation test']
df['test correlation all tracks model'] = df_correlation_test_alltracksmodel['correlation test all tracks model']
# df['valid correlation'] = df_correlation_valid['correlation valid']
# df['train correlation'] = df_correlation_train['correlation train']

print(f'mean correlation score test: {df["test correlation"].mean(axis=0):.4f}')
print(f'mean correlation score test all tracks model: {df["test correlation all tracks model"].mean(axis=0):.4f}')

print(df)

plt.figure(1)
ax = sns.boxplot(data = df[[ 'test correlation all tracks model', 'test correlation']], showmeans = True)
ax.set_xticklabels(['All tracks model', 'New tracks model'])
plt.ylabel('Pearson Correlation Coefficient')
plt.title(f'Test set correlation for 19 tracks')
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_newtracks/dnn_newtracks_boxplot_test_newvsall_corr.png', bbox_inches='tight')

for key, value in df['assay type'].value_counts().to_dict().items():
    df_subset = df[df['assay type'] == key]
    print(key, value)
    plt.figure()
    plt.title(f'Assay type: {key}')
    plt.axline((0, 0), (1, 1), linewidth=0.5, color='k', linestyle = 'dashed')
    sns.scatterplot(data = df_subset, x = 'test correlation', y = 'test correlation all tracks model', color = 'k')
    plt.ylabel('Test correlation new tracks model')
    plt.xlabel('Test correlation all tracks model')
    plt.savefig(f'/exports/humgen/idenhond/projects/enformer/correlation/Plots/dnn_newtracks/dnn_newtracks_scatterplot_test_corr_newvsall_{key}.png', bbox_inches='tight')


exit()
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


