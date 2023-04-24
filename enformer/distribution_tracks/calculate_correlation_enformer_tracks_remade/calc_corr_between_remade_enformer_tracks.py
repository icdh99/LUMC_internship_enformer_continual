import torch
from torchmetrics import PearsonCorrCoef

# calculate correlation between 3 tracks from enformer that I remade

pearson = PearsonCorrCoef()

corr_track1 = 0
corr_track2 = 0
corr_track3 = 0

for i in range(1, 34021+1):
    print(i)
    # load target for sequence i
    target_enformer = torch.load(f'/exports/humgen/idenhond/data/Enformer_train/Enformer_train_targets/targets_seq{i}.pt').squeeze()
    # print(target_enformer.shape)    # torch.Size([1, 896, 5313])
    # target_mine = torch.load(f'/exports/humgen/idenhond/data/Enformer_train/Enformer_train_targets_newtracks2703/targets_seq{i}.pt').squeeze()
    target_mine = torch.load(f'/exports/humgen/idenhond/data/Enformer_train/Enformer_train_3tracks_remade/targets_seq{i}.pt').squeeze()
    # print(target_mine.shape)    # torch.Size([1, 896, 5313])

    # calculate correlation for track 1
    track1_enf = target_enformer[:, 0]
    # print(track1_enf.shape)
    track1_mine = target_mine[:, 0]
    # print(track1_enf[:10])
    # print(track1_mine[:10])
    # print(track1_mine.shape)
    # print(pearson(track1_enf, track1_mine))
    corr_track1 += pearson(track1_enf, track1_mine)

    # calculate correlation for track 2
    track2_enf = target_enformer[:, 1454]
    # print(track2_enf.shape)
    track2_mine = target_mine[:, 1]
    # print(track2_mine.shape)
    # print(pearson(track2_enf, track2_mine))
    # print(track2_enf[:10])
    # print(track2_mine[:10])
    corr_track2 += pearson(track2_enf, track2_mine)

    # calculate correlation for track 3
    track3_enf = target_enformer[:, 1028]
    # print(track3_enf.shape)
    track3_mine = target_mine[:, 2]
    # print(track3_enf[:10])
    # print(track3_mine[:10])
    # print(track3_mine.shape)
    # print(pearson(track3_enf, track3_mine))
    corr_track3 += pearson(track3_enf, track3_mine)

    # if i == 1: break

final_corr_track1 = corr_track1 / i
final_corr_track2 = corr_track2 / i
final_corr_track3 = corr_track3 / i

print(f'final correlation for track 1 calculated over {i} sequences: {final_corr_track1}')
print(f'final correlation for track 2 calculated over {i} sequences: {final_corr_track2}')
print(f'final correlation for track 3 calculated over {i} sequences: {final_corr_track3}')