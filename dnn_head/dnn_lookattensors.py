import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device (set to GPU if available):', device)

### show first tensor of input embeddings X
t_input = torch.load('tensor_embeddingsvalidation_100.pt', map_location=torch.device(device))
print(f'first tensor of embeddings input X')
print(f'shape: {t_input.shape}')    # shape: torch.Size([100, 896, 3072])
print(f'first tensor:')
print(f'{t_input[0]}')

# show first tensor of target output X
t_output = torch.load('tensor_targetvalidation_100.pt', map_location=torch.device(device))
print(f'\nfirst tensor of target output Y')
print(f'shape: {t_output.shape}')   # shape: torch.Size([100, 896, 5313])
print(f'first tensor:')
print(f'{t_output[0]}')