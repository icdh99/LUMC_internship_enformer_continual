import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device (set to GPU if available):', device)

t_input1 = torch.load('/exports/humgen/idenhond/data/Enformer_train/embeddings_train_pretrainedmodel_one.pt', map_location=torch.device(device))
t_input2 = torch.load('/exports/humgen/idenhond/data/Enformer_train/embeddings_train_pretrainedmodel_two.pt', map_location=torch.device(device))
t_input3 = torch.load('/exports/humgen/idenhond/data/Enformer_train/embeddings_train_pretrainedmodel_three.pt', map_location=torch.device(device))
t_input4 = torch.load('/exports/humgen/idenhond/data/Enformer_train/embeddings_train_pretrainedmodel_four.pt', map_location=torch.device(device))

print(f'shape tensor 1: {t_input1.shape}')
print(f'shape tensor 2: {t_input2.shape}')
print(f'shape tensor 3: {t_input3.shape}')
print(f'shape tensor 4: {t_input4.shape}')

t = torch.cat([t_input1, t_input2, t_input3, t_input4], dim = 0)

print(f'shape of final tensor: {t.shape}')

torch.save(t, '/exports/humgen/idenhond/data/Enformer_train/embeddings_train.pt')
