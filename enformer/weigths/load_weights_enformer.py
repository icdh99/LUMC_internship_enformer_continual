from enformer_pytorch import Enformer
from collections import OrderedDict
import torch

model = Enformer.from_pretrained("EleutherAI/enformer-official-rough")

weights = model.state_dict()

print(type(weights)) # ordered dict
# print(weights)

with open('weights_enformer_keylist.csv', 'w') as f:
    f.truncate(0)
    for key, value in weights.items():
        print(key, value.shape) # value is a torch tensor
        f.write(' '.join([key, str(value.shape), '\n']))
    
# print(weights['_heads.human.0.weight'])

# torch.save(weights['_heads.human.0.weight'], 'heads_human_0_weight.pt')
torch.save(weights['_heads.human.0.bias'], 'heads_human_0_bias.pt')