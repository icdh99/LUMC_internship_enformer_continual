import torch

# map_location=torch.device('cpu')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
t = torch.load('/exports/humgen/idenhond/enformer_dev/enformer-pytorch/data/test-sample.pt', map_location=torch.device('cpu'))

print(t)
seq = t['sequence']

print(type(seq))
print(seq.shape)
print(seq.dtype)
print(seq.device)

target = t['target']

print(type(target))
print(target.shape)
print(target.dtype)
print(target.device)