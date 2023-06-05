import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd

w = torch.load(f'/exports/humgen/idenhond/projects/enformer/weigths/heads_human_0_weight.pt')

print(w.shape)

print(w[0])
print(w[0].shape)

w = w.detach().numpy()
print(type(w))
print(w.shape)

w = pd.DataFrame(w)
print(w)
print(w.shape)

df_reset = w.reset_index() 
print(df_reset)

df_melted = pd.melt(df_reset, id_vars='index', var_name='x', value_name='value')
df_melted['x'] = pd.to_numeric(df_melted['x'])

print(df_melted)

plt.figure()
sns.lineplot(data=df_melted, x='x', y='value', hue='index', legend=False)
plt.xlabel('X')
plt.ylabel('Value')
plt.title('Weights for _heads.human.0.weight (shape 5313 x 3072)')
plt.savefig('weights_heads_human_0.png')
plt.close()