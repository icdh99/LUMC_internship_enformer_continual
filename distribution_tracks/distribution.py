"""
Outputs of dnn head model
/exports/archive/hg-funcgenom-research/idenhond/Enformer_test/Enformer_test_output_newmodel/output_test.pt
/exports/archive/hg-funcgenom-research/idenhond/Enformer_validation/Enformer_validation_output_newmodel/output_validation.pt

Outputs of enformer model (zelf gerund)
niet opgeslagen
"""
from datetime import datetime
start = datetime.now()
import torch
import matplotlib.pyplot as plt

torch.Tensor.ndim = property(lambda self: len(self.shape))

# ### output test linear & log scale
# path_test = '/exports/archive/hg-funcgenom-research/idenhond/Enformer_test/Enformer_test_output_newmodel/output_test.pt'
# test = torch.load(path_test, map_location='cpu')
# print(test.shape)   # torch.Size([1937, 896, 5313])
# print(len(test)) # 1937
# # for i in range(5313):
# #     plt.plot(test[0, :, i], linewidth = 0.01, color = 'k')
# # plt.yscale('linear')
# # plt.savefig('test_output_firstseq_alltracks_linear.png')
# # for i in range(5313):
# #     plt.plot(test[0, :, i], linewidth = 0.01, color = 'k')
# # plt.yscale('log')
# # plt.savefig('test_output_firstseq_alltracks_log.png')

# plt.figure()
# plt.imshow(test[0, :, :].numpy(), cmap = 'hot', interpolation = 'nearest')
# plt.savefig('test_output_firstseq_alltracks_heatmap.png')
# del test

path_valid = '/exports/archive/hg-funcgenom-research/idenhond/Enformer_validation/Enformer_validation_output_newmodel/output_validation.pt'
valid = torch.load(path_valid, map_location='cpu')
print(valid.shape)
print(len(valid))
### output valid linear & log scale
for i in range(5313):
    plt.plot(valid[0, :, i], linewidth = 0.01, color = 'k')
plt.yscale('linear')
plt.savefig('valid_output_firstseq_alltracks_linear.png')
for i in range(5313):
    plt.plot(valid[0, :, i], linewidth = 0.01, color = 'k')
plt.yscale('symlog')
plt.savefig('valid_output_firstseq_alltracks_log.png')

# for i in range(1, 2213+1):
#     t = torch.load(f'/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_targets_perseq/targets_seq{i}.pt', map_location='cpu')
#     print(t.shape)
#     break

print(f'Time: {datetime.now() - start}') 