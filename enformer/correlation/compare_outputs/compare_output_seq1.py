import torch
"""
compare output of test sequence 1 
- own output from /get_output_emb/store_tensors_test.py
- output from /correlation/evaluate_correlation.py
"""

filepath = '/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output/output_test.pt'
tensor_out_old = torch.load(filepath, map_location='cpu')
print(f'device of output tensor: {tensor_out_old.device}')
print(f'shape of output tensor: {tensor_out_old.shape}\n')

print('\n')
for t in range(1, 1937+1):
    print(f'{t} new')
    filepath = f'/exports/humgen/idenhond/data/Enformer_test/Enformer_test_output_newmodel/output_seq{t}.pt'
    tensor_out_new = torch.load(filepath, map_location='cpu')
    # print(f'device of output tensor: {tensor_out_new.device}')
    # print(f'shape of output tensor: {tensor_out_new.shape}\n')

    pred_new = torch.unsqueeze(tensor_out_new, 0)
    for i in range(0, 1937):
        pred_old = torch.unsqueeze(tensor_out_old[i], 0)
        # print(f'{i} old')
        eq = torch.equal(pred_old, pred_new)
        # print(f'is pred_old equal to pred new model: {eq}')
        if eq:
            print(f'{t} old, {i} new\n {pred_old}\n{pred_new}')



