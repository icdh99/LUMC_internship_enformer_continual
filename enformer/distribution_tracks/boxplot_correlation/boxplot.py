import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.patches import PathPatch

def adjust_box_widths(g, fac):
    """
    Adjust the widths of a seaborn-generated boxplot.
    """

    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])

np.set_printoptions(linewidth=400) 

filename_targets = '/exports/humgen/idenhond/projects/enformer/distribution_tracks/boxplot_correlation/final_correlation_matrix_alltracks_testset_targets.csv'
targets = np.loadtxt(filename_targets, delimiter=',')
filename_outputs = '/exports/humgen/idenhond/projects/enformer/distribution_tracks/boxplot_correlation/final_correlation_matrix_alltracks_testset_outputs.csv'
outputs = np.loadtxt(filename_outputs, delimiter=',')
filename_enformer_outputs = '/exports/humgen/idenhond/projects/enformer/distribution_tracks/boxplot_correlation/final_correlation_matrix_alltracks_testset_outputs_enformer.csv'
outputs_enformer = np.loadtxt(filename_enformer_outputs, delimiter=',')
print(outputs_enformer.shape)

# targets 1 category
targets_dnase = targets[:684, :684]
targets_chip = targets[684:4675, 684:4675]
targets_cage = targets[4675:5313, 4675:5313]
targets_new = targets[5313:5340, 5313:5340]
targets_new_histone = targets[5313:5322]
targets_new_tf = targets[5322:5331]
targets_new_dnase = targets[5331:5340]

# targets 2 categories
targets_dnase_vs_chip = targets[:684, 684:4675]
targets_dnase_vs_cage = targets[:684, 4675:5313]
targets_cage_vs_chip = targets[4675:5313, 684:4675]
targets_new_vs_dnase = targets[5313:5340, :684]
targets_new_vs_chip = targets[5313:5340, 684:4675]
targets_new_vs_cage = targets[5313:5340, 4675:5313]
targets_new_dnase_vs_dnase = targets[5331:5340, :684]
targets_new_chip_vs_chip = targets[5313:5331, 684:4675] # both histone & tf chip

# outputs 1 category
outputs_dnase = outputs[:684, :684]
outputs_chip = outputs[684:4675, 684:4675]
outputs_cage = outputs[4675:5313, 4675:5313]
outputs_new = outputs[5313:5340, 5313:5340]
outputs_new_histone = outputs[5313:5322]
outputs_new_tf = outputs[5322:5331]
outputs_new_dnase = outputs [5331:5340]
outputs_enformer_dnase = outputs_enformer[:684, :684]
outputs_enformer_chip = outputs_enformer[684:4675, 684:4675]
outputs_enformer_cage = outputs_enformer[4675:5313, 4675:5313]

# outputs 2 categories
outputs_dnase_vs_chip = outputs[:684, 684:4675]
outputs_dnase_vs_cage = outputs[:684, 4675:5313]
outputs_cage_vs_chip = outputs[4675:5313, 684:4675]
outputs_new_vs_dnase = outputs[5313:5340, :684]
outputs_new_vs_chip = outputs[5313:5340, 684:4675]
outputs_new_vs_cage = outputs[5313:5340, 4675:5313]
outputs_new_dnase_vs_dnase = outputs[5331:5340, :684]
outputs_new_chip_vs_chip = outputs[5313:5331, 684:4675] # both histone & tf chip

outputs_enformer_dnase_vs_chip = outputs_enformer[:684, 684:4675]
outputs_enformer_dnase_vs_cage = outputs_enformer[:684, 4675:5313]
outputs_enformer_cage_vs_chip = outputs_enformer[4675:5313, 684:4675]

print(targets_dnase.shape)
print(targets_chip.shape)
print(targets_cage.shape)
print(targets_new.shape)
print(targets_dnase_vs_chip.shape)

values_targets_dnase = targets_dnase[np.tril_indices(684, k=-1)].tolist()
values_targets_chip = targets_chip[np.tril_indices(3991, k=-1)].tolist()
values_targets_cage = targets_cage[np.tril_indices(638, k=-1)].tolist()
values_targets_newtracks = targets_new[np.tril_indices(27, k=-1)].tolist()
values_targets_newtracks_histone = targets_new_histone[np.tril_indices(9, k=-1)].tolist()
values_targets_newtracks_tf = targets_new_tf[np.tril_indices(9, k=-1)].tolist()
values_targets_newtracks_dnase = targets_new_dnase[np.tril_indices(9, k=-1)].tolist()

values_targets_dnase_vs_chip = [item for sublist in targets_dnase_vs_chip.tolist() for item in sublist]
values_targets_dnase_vs_cage = [item for sublist in targets_dnase_vs_cage.tolist() for item in sublist]
values_targets_cage_vs_chip = [item for sublist in targets_cage_vs_chip.tolist() for item in sublist]
values_targets_new_vs_dnase = [item for sublist in targets_new_vs_dnase.tolist() for item in sublist]
values_targets_new_vs_chip = [item for sublist in targets_new_vs_chip.tolist() for item in sublist]
values_targets_new_vs_cage = [item for sublist in targets_new_vs_cage.tolist() for item in sublist]

values_outputs_dnase = outputs_dnase[np.tril_indices(684, k=-1)].tolist()
values_outputs_chip = outputs_chip[np.tril_indices(3991, k=-1)].tolist()
values_outputs_cage = outputs_cage[np.tril_indices(638, k=-1)].tolist()
values_outputs_newtracks = outputs_new[np.tril_indices(27, k=-1)].tolist()
values_outputs_newtracks_histone = outputs_new_histone[np.tril_indices(9, k=-1)].tolist()
values_outputs_newtracks_tf = outputs_new_tf[np.tril_indices(9, k=-1)].tolist()
values_outputs_newtracks_dnase = outputs_new_dnase[np.tril_indices(9, k=-1)].tolist()

values_outputs_dnase_vs_chip = [item for sublist in outputs_dnase_vs_chip.tolist() for item in sublist]
values_outputs_dnase_vs_cage = [item for sublist in outputs_dnase_vs_cage.tolist() for item in sublist]
values_outputs_cage_vs_chip = [item for sublist in outputs_cage_vs_chip.tolist() for item in sublist]
values_outputs_new_vs_dnase = [item for sublist in outputs_new_vs_dnase.tolist() for item in sublist]
values_outputs_new_vs_chip = [item for sublist in outputs_new_vs_chip.tolist() for item in sublist]
values_outputs_new_vs_cage = [item for sublist in outputs_new_vs_cage.tolist() for item in sublist]

values_outputs_enformer_dnase = outputs_enformer_dnase[np.tril_indices(684, k=-1)].tolist()
values_outputs_enformer_chip = outputs_enformer_chip[np.tril_indices(3991, k=-1)].tolist()
values_outputs_enformer_cage = outputs_enformer_cage[np.tril_indices(638, k=-1)].tolist()

values_outputs_enformer_dnase_vs_chip = [item for sublist in outputs_enformer_dnase_vs_chip.tolist() for item in sublist]
values_outputs_enformer_dnase_vs_cage = [item for sublist in outputs_enformer_dnase_vs_cage.tolist() for item in sublist]
values_outputs_enformer_cage_vs_chip = [item for sublist in outputs_enformer_cage_vs_chip.tolist() for item in sublist]

# data = {"List": ["DNase, ATAC"] * len(values_targets_dnase) * 3 + ["ChIP"] * len(values_targets_chip) *3 + ["CAGE"] * len(values_targets_cage) *3  + ["DNase, ATAC vs ChIP"] * len(values_targets_dnase_vs_chip) *3 + ["DNase, ATAC vs CAGE"] * len(values_targets_dnase_vs_cage) *3 + ["CAGE vs ChIP"] * len(values_targets_cage_vs_chip) *3,
#         "Value": values_targets_dnase + values_outputs_dnase + values_outputs_enformer_dnase + values_targets_chip + values_outputs_chip + values_outputs_enformer_chip + values_targets_cage + values_outputs_cage + values_outputs_enformer_cage + values_targets_dnase_vs_chip + values_outputs_dnase_vs_chip + values_outputs_enformer_dnase_vs_chip + values_targets_dnase_vs_cage + values_outputs_dnase_vs_cage + values_outputs_enformer_dnase_vs_cage + values_targets_cage_vs_chip + values_outputs_cage_vs_chip + values_outputs_enformer_cage_vs_chip,
#         "Type": ["Target"] * len(values_targets_dnase) + ["Output Human Head"] * len(values_outputs_dnase) + ["Output Enformer-pytorch"] * len(values_outputs_dnase) + ["Target"] * len(values_targets_chip)+ ["Output Human Head"] * len(values_targets_chip) + ["Output Enformer-pytorch"] * len(values_targets_chip) + ["Target"] * len(values_targets_cage)+ ["Output Human Head"] * len(values_targets_cage) + ["Output Enformer-pytorch"] * len(values_targets_cage)  + ["Target"] * len(values_targets_dnase_vs_chip)+ ["Output Human Head"] * len(values_targets_dnase_vs_chip) + ["Output Enformer-pytorch"] * len(values_targets_dnase_vs_chip) + ["Target"] * len(values_targets_dnase_vs_cage)+ ["Output Human Head"] * len(values_targets_dnase_vs_cage) + ["Output Enformer-pytorch"] * len(values_targets_dnase_vs_cage) + ["Target"] * len(values_targets_cage_vs_chip)+ ["Output Human Head"] * len(values_targets_cage_vs_chip) + ["Output Enformer-pytorch"] * len(values_targets_cage_vs_chip) }
# df = pd.DataFrame(data)

# print(len(df['List']))
# print(len(df['Value']))
# print(len(df['Type']))
# print(df.tail)

# PROPS = {
#     'boxprops':{'facecolor':'none', 'edgecolor':'black'},
#     'medianprops':{'color':'black'},
#     'whiskerprops':{'color':'black'},
#     'capprops':{'color':'black'}
# }

# plt.figure(figsize = (5.0, 5.8)) #width, height (6.4 4.8 default)
# flierprops = dict(marker='o', markerfacecolor='None', markersize=0.3,  markeredgecolor='black')
# ax = sns.boxplot(data = df, x = 'Value', y = 'List', hue = 'Type', linewidth = 0.8, palette = sns.color_palette("Paired"), flierprops=flierprops)
# sns.despine(top=True, right=True, left=False, bottom=False)
# ax.tick_params(axis='both', which='major', labelsize=7)
# # plt.title('Correlation between true and predicted genomic tracks per assay type')
# plt.xlabel('Pearson Correlation Coefficient', fontsize = 7)
# plt.ylabel(None)
# plt.xticks([-0.50, 0, 0.50, 1], [-0.50, 0, 0.50, 1])
# plt.legend(bbox_to_anchor=(0.01, 1), loc='upper left', prop={'size': 6})
# plt.savefig('boxplot_enformer_tracks.png', bbox_inches = 'tight', dpi = 300)
# plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig1_correlation_dnnhead/Boxplot_enformer_tracks.png', bbox_inches = 'tight', dpi = 300)
# plt.close()
# fig_width, fig_height = plt.gcf().get_size_inches()
# print(fig_width, fig_height)

data = {"List": ['Histone ChIP'] * len(values_outputs_newtracks_histone) * 2 + ['TF ChIP'] * len(values_outputs_newtracks_tf) * 2  + ['DNase'] * len(values_outputs_newtracks_dnase) * 2  ,
        "Value": values_targets_newtracks_histone + values_outputs_newtracks_histone + values_targets_newtracks_tf + values_outputs_newtracks_tf + values_targets_newtracks_dnase + values_outputs_newtracks_dnase,
        "Type": ['Target'] * len(values_outputs_newtracks_histone) + ['Output'] * len(values_outputs_newtracks_histone) +  ['Target'] * len(values_outputs_newtracks_tf) + ['Output'] * len(values_outputs_newtracks_tf) + ['Target'] * len(values_outputs_newtracks_dnase) + ['Output'] * len(values_outputs_newtracks_dnase)
}

df = pd.DataFrame(data)

palette_paired= sns.color_palette("Paired")
palette = sns.color_palette([palette_paired[4], palette_paired[5]])
fig = plt.figure(figsize = (5.0, 5.8))
plt.tight_layout()
flierprops = dict(marker='o', markerfacecolor='None', markersize=1,  markeredgecolor='black')
sns.boxplot(data = df, x = 'List', y = 'Value', hue = 'Type', palette = palette, flierprops=flierprops)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.ylabel('Pearson Correlation Coefficient', fontsize = 10)
plt.legend(title = None)
plt.xlabel(None)
adjust_box_widths(fig, 0.9)
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig2_newtracks/Boxplot_newtracks.png', bbox_inches = 'tight', dpi = 300)
plt.savefig('boxplot_newtracks_perassaytype.png', bbox_inches = 'tight')
plt.close()


# # new tracks outputs & targets
# data = {"List": ["NEW"] * len(values_targets_newtracks) *2 + ["NEW vs DNASE"] * len(values_targets_new_vs_dnase)*2  + ["NEW vs ChIP"] * len(values_targets_new_vs_chip)*2  + ["NEW vs CAGE"] * len(values_targets_new_vs_cage)*2 , 
#         "Value": values_targets_newtracks + values_outputs_newtracks + values_targets_new_vs_dnase + values_outputs_new_vs_dnase + values_targets_new_vs_chip + values_outputs_new_vs_chip + values_targets_new_vs_cage + values_outputs_new_vs_cage,
#         "Type": ["TARGET"] * len(values_targets_newtracks) + ["OUTPUT"] * len(values_targets_newtracks) + ["TARGET"] * len(values_targets_new_vs_dnase) + ["OUTPUT"] * len(values_targets_new_vs_dnase) + ["TARGET"] * len(values_targets_new_vs_chip) + ["OUTPUT"] * len(values_targets_new_vs_chip) + ["TARGET"] * len(values_targets_new_vs_cage) + ["OUTPUT"] * len(values_targets_new_vs_cage)}
# df = pd.DataFrame(data)

# plt.figure()
# flierprops = dict(marker='o', markerfacecolor='None', markersize=0.5,  markeredgecolor='black')
# sns.boxplot(data = df, x = 'List', y = 'Value', hue = 'Type', palette = sns.color_palette("Paired"), flierprops=flierprops)
# plt.xticks(rotation = 45)
# plt.title('Correlation between true and predicted genomic tracks per assay type')
# plt.xlabel('Assay type comparison')
# plt.ylabel('Correlation')
# plt.savefig('boxplot_newtracks.png', bbox_inches = 'tight')
# plt.close()

