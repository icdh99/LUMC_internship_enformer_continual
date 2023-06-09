import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns
import pandas as pd
import random
random.seed(21)
"""
Plot Enformer track prediction vs target
"""

# get 20 highest correlation tracks on the test set
input_file = '/exports/humgen/idenhond/data/basenji_preprocess/output_tfr_27newtracks_human/targets.txt'
df = pd.read_csv(input_file, sep = '\t')


df_correlation_test = pd.read_csv('/exports/humgen/idenhond/data/evaluate_correlation/correlation_per_track_test_newtracks_2404.csv', index_col = 0, names = ['correlation test']).tail(-1)
df['test correlation'] = df_correlation_test['correlation test']

print(df)


# get random number of sequences
seq_numbers = [random.randint(0, 1937) for _ in range(20)]
df_seq_nrs = pd.DataFrame(seq_numbers, columns = ['seq_nr'])
df_test = pd.read_csv('/exports/humgen/idenhond/data/DAR/Enformer_test/enformer_test.bed', sep = '\t', names = ['chr', 'start', 'stop'])
df_test['interval'] = df_test['chr'] + ':' + df_test['start'].astype(str) + '-' + df_test['stop'].astype(str)
df_test['index'] = df_test.index
df_seq_nrs = df_seq_nrs.merge(df_test, left_on = 'seq_nr', right_on='index', how = 'inner')
print(df_seq_nrs) #  seq_nr    chr      start       stop                  interval  index

output_folder = '/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output_newtracks_2404'
target_folder = '/exports/humgen/idenhond/data/Enformer_test/Newtracks_2404_test_targets'

def plot_tracks(tracks, interval, track_nr, seq_nr, start, stop, height=2):
  with sns.plotting_context("talk"):
    print(len(tracks))
    fig, axes = plt.subplots(len(tracks), 1, figsize=(20, height * len(tracks)), sharex=True, constrained_layout = True)
    fig_width, fig_height = plt.gcf().get_size_inches()
    # print(fig_width, fig_height)
    plt.tight_layout()
    for i, (ax, (title, y)) in enumerate(zip(axes, tracks.items())):
      # print(i, title) 
      if i == 0 or i == 1:
          color = 'steelblue'
      elif i == 2 or i == 3:
         color = 'sandybrown'
      elif i == 4 or i == 5:
          color = 'forestgreen'

      ax.fill_between(np.linspace(start, stop, num=len(y)), y, color = color) 
      ax.set_title(title, y = 1)
      sns.despine(top=True, right=True, bottom=True)
    ax.set_xlabel(f'{interval}', labelpad=20) 
    plt.savefig(f'Plots_Fig2/Track{track_nr}_seq{seq_nr}.png', bbox_inches = 'tight', dpi = 300)
    plt.close()
    
for index, row in df_seq_nrs.iterrows():
    # print(index, row)
    seq_nr = row.seq_nr
    interval = row.interval
    start = row.start + 8192
    stop = row.stop - 8192
    target_tensor = torch.load(f'{target_folder}/targets_seq{seq_nr}.pt', map_location=torch.device('cpu')).squeeze()
    output_tensor = torch.load(f'{output_folder}/output_seq{seq_nr}.pt', map_location=torch.device('cpu')).squeeze() 
    # print(target_tensor.shape)
    # print(output_tensor.shape)
    track_nr = 'combined'
    tracks = {'DNASE:Homo sapiens suppressor macrophage male adult (24 years) observed': target_tensor[:, 25],
            'DNASE:Homo sapiens suppressor macrophage male adult (24 years) prediction': output_tensor[:, 25],
            'CHIP:CTCF:Homo sapiens heart left ventricle tissue male adult (69 years) observed': target_tensor[:, 15], 
            'CHIP:CTCF:Homo sapiens heart left ventricle tissue male adult (69 years) prediction': output_tensor[:, 15],
            'CHIP:H3K4me3:Homo sapiens middle frontal area 46 tissue male adult (83 years) observed': target_tensor[:, 3], 
            'CHIP:H3K4me3:Homo sapiens middle frontal area 46 tissue male adult (83 years) prediction': output_tensor[:, 3]}

    print(len(tracks))
    plot_tracks(tracks, interval, track_nr, seq_nr, start, stop)

    
# seq_nr = 1623 
   