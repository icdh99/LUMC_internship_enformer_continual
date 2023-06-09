import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns
import pandas as pd
import random
random.seed(18)
"""
Plot Enformer track prediction vs target
"""

# get 20 highest correlation tracks on the test set
input_file = '/exports/humgen/idenhond/data/Basenji/human-targets.txt'
df = pd.read_csv(input_file, sep = '\t')
df[['assay type', 'description2']] = df.description.str.split(':', n = 1, expand = True) # make new column for assay type
def f(row):
    if row['assay type'] == 'CHIP':
        if any(row['description2'].startswith(x) for x in ['H2AK', 'H2BK', 'H3K', 'H4K']): val = 'ChIP Histone'
        else: val = 'ChIP TF'
    elif row['assay type'] == 'DNASE' or row['assay type'] == 'ATAC': val = 'DNASE_ATAC'
    else: val = row['assay type']
    return val
df['assay type split ChIP'] = df.apply(f, axis=1)
df_correlation_test = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_dnn_head.csv', index_col = 0, names = ['correlation test']).tail(-1)
df['test correlation'] = df_correlation_test['correlation test']

print(f'20 highest test correlations:')
df_top20 = df.nlargest(20, 'test correlation')
print(df_top20)

print(f'five highest DNase/ATAC tracks:')
df_dnase_atac_top5 = df[df['assay type split ChIP'] == 'DNASE_ATAC'].nlargest(5, 'test correlation')
print(df_dnase_atac_top5[['index', 'description', 'test correlation']])



# get random number of sequences
seq_numbers = [random.randint(0, 1937) for _ in range(20)]
df_seq_nrs = pd.DataFrame(seq_numbers, columns = ['seq_nr'])
df_test = pd.read_csv('/exports/humgen/idenhond/data/DAR/Enformer_test/enformer_test.bed', sep = '\t', names = ['chr', 'start', 'stop'])
df_test['interval'] = df_test['chr'] + ':' + df_test['start'].astype(str) + '-' + df_test['stop'].astype(str)
df_test['index'] = df_test.index
df_seq_nrs = df_seq_nrs.merge(df_test, left_on = 'seq_nr', right_on='index', how = 'inner')
print(df_seq_nrs) #  seq_nr    chr      start       stop                  interval  index

output_folder = '/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output'
target_folder = '/exports/humgen/idenhond/data/Enformer_test/Enformer_test_targets_perseq'

def plot_tracks(tracks, interval, track_nr, seq_nr, start, stop, height=2):
  with sns.plotting_context("talk"):
    fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True, constrained_layout = True)
    fig_width, fig_height = plt.gcf().get_size_inches()
    # print(fig_width, fig_height)
    plt.tight_layout()
    for i, (ax, (title, y)) in enumerate(zip(axes, tracks.items())):
    #   print(i, title) 
      if i == 0 or i == 1:
          color = 'steelblue'
      else:
          color = 'sandybrown'
      ax.fill_between(np.linspace(start, stop, num=len(y)), y, color = color) 
      ax.set_title(title, y = 1)
      sns.despine(top=True, right=True, bottom=True)
    ax.set_xlabel(f'{interval}', labelpad=20) 
    plt.savefig(f'Track{track_nr}_seq{seq_nr}.png', bbox_inches = 'tight', dpi = 300)
    plt.close()
    
for index, row in df_seq_nrs.iterrows():
    # print(index, row)
    seq_nr = row.seq_nr
    interval = row.interval
    start = row.start + 8192
    stop = row.stop - 8192
    target_tensor = torch.load(f'{target_folder}/targets_seq{seq_nr}.pt', map_location=torch.device('cpu')).squeeze()
    output_tensor = torch.load(f'{output_folder}/output_seq{seq_nr}.pt', map_location=torch.device('cpu')).squeeze() 
    # track_nr = 2136
    # tracks = {'ZFX:HEK293T target': target_tensor[:, track_nr],
    #         'ZFX:HEK293T prediction': output_tensor[:, track_nr]}
    # plot_tracks(tracks, interval, track_nr, seq_nr, start, stop)
    # track_nr = 1235
    # tracks = {'H3K4me3:skeletal muscle cell target': target_tensor[:, track_nr],
    #         'H3K4me3:skeletal muscle cell prediction': output_tensor[:, track_nr]}
    # plot_tracks(tracks, interval, track_nr, seq_nr, start, stop)
    # track_nr = 1161
    # tracks = {'H3K4me3:astrocyte of the cerebellum target': target_tensor[:, track_nr],
    #         'H3K4me3:astrocyte of the cerebellum prediction': output_tensor[:, track_nr]}
    # plot_tracks(tracks, interval, track_nr, seq_nr, start, stop)
    # track_nr = 1169
    # tracks = {'H3K4me3:cardiac muscle cell target': target_tensor[:, track_nr],
    #         'H3K4me3:cardiac muscle cell prediction': output_tensor[:, track_nr]}
    # plot_tracks(tracks, interval, track_nr, seq_nr, start, stop)
    # track_nr = 1988
    # tracks = {'H3K4me3:foreskin keratinocyte male newborn target': target_tensor[:, track_nr],
    #         'H3K4me3:foreskin keratinocyte male newborn prediction': output_tensor[:, track_nr]}
    # plot_tracks(tracks, interval, track_nr, seq_nr, start, stop)
    # track_nr = 1167
    # tracks = {'H3K4me3:cardiac fibroblast female target': target_tensor[:, track_nr],
    #         'H3K4me3:cardiac fibroblast female prediction': output_tensor[:, track_nr]}
    # plot_tracks(tracks, interval, track_nr, seq_nr, start, stop)
    # track_nr = 1194
    # tracks = {'H3K4me3:fibroblast of pulmonary artery target': target_tensor[:, track_nr],
    #         'H3K4me3:fibroblast of pulmonary artery prediction': output_tensor[:, track_nr]}
    # plot_tracks(tracks, interval, track_nr, seq_nr, start, stop)
    # track_nr = 1164
    # tracks = {'H3K4me3:brain microvascular endothelial cell target': target_tensor[:, track_nr],
    #         'H3K4me3:brain microvascular endothelial cell prediction': output_tensor[:, track_nr]}
    # plot_tracks(tracks, interval, track_nr, seq_nr, start, stop)
    # track_nr = 270
    # tracks = {'DNASE:stomach female embryo (98 days) cell target': target_tensor[:, track_nr],
    #         'DNASE:stomach female embryo (98 days) cell prediction': output_tensor[:, track_nr]}
    # plot_tracks(tracks, interval, track_nr, seq_nr, start, stop)
    # track_nr = 60
    # tracks = {'DNASE:B cell female adult (43 years) target': target_tensor[:, track_nr],
    #         'DNASE:B cell female adult (43 years) prediction': output_tensor[:, track_nr]}
    # plot_tracks(tracks, interval, track_nr, seq_nr, start, stop)
    # track_nr = 564
    # tracks = {'DNASE:kidney embryo (59 days) and female embryo (59 days) target': target_tensor[:, track_nr],
    #         'DNASE:kidney embryo (59 days) and female embryo (59 days) prediction': output_tensor[:, track_nr]}
    # plot_tracks(tracks, interval, track_nr, seq_nr, start, stop)
    # track_nr = 460
    # tracks = {'DNASE:renal pelvis male embryo (91 day) target': target_tensor[:, track_nr],
    #         'DNASE:renal pelvis male embryo (91 day) prediction': output_tensor[:, track_nr]}
    # plot_tracks(tracks, interval, track_nr, seq_nr, start, stop)
    # track_nr = 571
    # tracks = {'DNASE:femur female embryo (98 days) target': target_tensor[:, track_nr],
    #         'DNASE:femur female embryo (98 days) prediction': output_tensor[:, track_nr]}
    # plot_tracks(tracks, interval, track_nr, seq_nr, start, stop)


target_tensor = torch.load(f'{target_folder}/targets_seq490.pt', map_location=torch.device('cpu')).squeeze()
output_tensor = torch.load(f'{output_folder}/output_seq490.pt', map_location=torch.device('cpu')).squeeze() 
track_nr = 'combine_dnase_chip'
interval = 'chr14:21684307-21798995'
start = 21684307 + 8192
stop = 21798995 - 8192
seq_nr = 490
tracks = {'DNASE:femur female embryo (98 days) observed': target_tensor[:, 571],
        'Prediction': output_tensor[:, 571],
        'ZFX:HEK293T observed': target_tensor[:, 2136],
        'Prediction': output_tensor[:, 2136]}
plot_tracks(tracks, interval, track_nr, seq_nr, start, stop)      