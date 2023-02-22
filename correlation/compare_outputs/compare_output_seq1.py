import torch
"""
compare output of test sequence 1 
- own output from /get_output_emb/store_tensors_test.py
- output from /correlation/evaluate_correlation.py
"""

## TODO: first look at own outputs from old model
print('tensor test seq 1 old model')
filepath = '/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output/output_test.pt'
tensor_out = torch.load(filepath)
print(f'device of output tensor: {tensor_out.device}')
print(f'shape of output tensor: {tensor_out.shape}\n')

pred = torch.unsqueeze(tensor_out[0], 0)

print(pred.device)
print(pred.shape)
print(pred)

torch.save(pred, 'outputownmodeltestseq1.pt')


print('\n')
## output seq1 from new model
print('tensor test seq 1 new model')
filepath = '/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output_newmodel/output_seq1.pt'
tensor_out = torch.load(filepath)
print(f'device of output tensor: {tensor_out.device}')
print(f'shape of output tensor: {tensor_out.shape}\n')

pred = torch.unsqueeze(tensor_out[0], 0)

print(pred.device)
print(pred.shape)
print(pred)