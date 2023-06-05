import torch
from model_class import model

"""
compare the weights before and after training of the human head model that is initialized with enformer weights
"""

w_before = torch.load(f'/exports/humgen/idenhond/projects/enformer/weigths/heads_human_0_weight.pt')
print(w_before.shape)

path = '/exports/humgen/idenhond/projects/enformer/dnn_head/train_dnn_head_init_weights/model_2023-05-20 07:29:47.870364/epoch=0-step=266-val_loss=inf.ckpt'
print(path)
model = model.load_from_checkpoint(path)
model.eval()

params = [param for param in model.parameters()]
print(params[0].shape)

print(w_before == params[0])
print(torch.equal(w_before,params[0]) )