import matplotlib.pyplot as plt
import seaborn as sns
import torch
import pandas as pd
import torch.nn as nn
import pytorch_lightning as pl

class model_dnnhead(pl.LightningModule):
	def __init__(self):
		# define model
		super(model_dnnhead, self).__init__()
		self.linear = nn.Linear(in_features = 3072, out_features = 5313, bias = True)
		self.softplus = nn.Softplus(beta = 1, threshold = 20)	# default values for nn.Softplus()
		self.lr = 1e-4
		self.loss = nn.PoissonNLLLoss()
		self.train_log = []
		self.save_hyperparameters()

	def forward(self, x):
		# define forward pass
		x = self.linear(x)
		x = self.softplus(x)
		return x

	def configure_optimizers(self):
		# define optimizer 
		return torch.optim.Adam(self.parameters(), lr = self.lr)

	def training_step(self, train_batch, batch_idx):
		# define training loop steps
		x, y = train_batch 
		logits = self.forward(x)
		loss = self.loss(logits, y)
		self.log("train_loss", loss, on_epoch=True, prog_bar=True)
		self.train_log.append(loss.cpu().detach().numpy())
		# tb_logger = self.logger.experiment
		# tb_logger.add_scalars("losses", {"train_loss": loss})
		self.logger.experiment.add_scalars('loss', {'train': loss},self.global_step) 
		return loss
	
	def test_step(self, batch, batch_idx):
		x, y = batch
		logits = self.forward(x)
		test_loss = self.loss(logits, y)
		self.log("test_loss", test_loss, on_epoch=True, prog_bar=True)
		return test_loss

	def validation_step(self, valid_batch, batch_idx):
		# define validation loop steps
		x, y = valid_batch
		logits = self.forward(x)
		val_loss = self.loss(logits, y)
		self.log("val_loss", val_loss, prog_bar=True)
		self.logger.experiment.add_scalars('loss', {'valid': val_loss},self.global_step)
		# self.logger.experiment.add_scalars("losses", {"val_loss": val_loss})

		return val_loss

	def predict_step(self, batch, batch_idx):
		x, y = batch
		return self(x), y

# dnn head weights
path = '/exports/humgen/idenhond/projects/enformer/dnn_head/dnn_head_train/model_2023-03-10 17:52:03.039827_v3/epoch=19-step=5320-val_loss=0.8.ckpt'
model = model_dnnhead.load_from_checkpoint(path)
model.eval()
params = [param for param in model.parameters()]
w = params[0].detach().numpy()
w = pd.DataFrame(w)
df_reset = w.reset_index() 
df_melted_humanhead = pd.melt(df_reset, id_vars='index', var_name='x', value_name='value human head')
df_melted_humanhead['x'] = pd.to_numeric(df_melted_humanhead['x'])

# enformer weights
w = torch.load(f'/exports/humgen/idenhond/projects/enformer/weigths/heads_human_0_weight.pt')
w = w.detach().numpy()
w = pd.DataFrame(w)
df_reset = w.reset_index() 
df_melted = pd.melt(df_reset, id_vars='index', var_name='x', value_name='value enformer')
df_melted['x'] = pd.to_numeric(df_melted['x'])
# print(df_melted.shape) # (16321536, 3). columns: index, x (nr of embedding), value

#### subset both weights to only specified tracks
df_melted = df_melted[df_melted['index'] <= 5000]
df_melted_humanhead = df_melted_humanhead[df_melted_humanhead['index'] <= 5000]

# concat and melt dataframe for seaborn plotting
df_concat = pd.concat([df_melted_humanhead['value human head'], df_melted['value enformer']], axis=1, keys=['value human head', 'value enformer'])
df_melt = pd.melt(df_concat )
print(df_melt)

# df_melt.astype({'value': 'float16'}).dtypes
# float32_cols = list(df_melt.select_dtypes(include='float32'))
# df_melt[float32_cols] = df_melt[float32_cols].astype('float16') # reduce memory for floats

df_melt = df_melt.dropna()
print(df_melt)

# see range of histogram for dnn head and enformer
print(df_melt[df_melt['variable'] == 'value human head'].min())
print(df_melt[df_melt['variable'] == 'value human head'].max())
print(df_melt[df_melt['variable'] == 'value enformer'].min())
print(df_melt[df_melt['variable'] == 'value enformer'].max())



legend_map = {'value human head': 'Human Head model',
              'value enformer': 'Enformer-pytorch'}

df_melt['variable'] = df_melt['variable'].map(legend_map)

plt.figure()
sns.displot(data=df_melt, x="value", hue = df_melt['variable'], bins=200, legend = True)
plt.yscale('log')
plt.xlabel('Weights (3072 * 5313)')
plt.ylabel('Count')
sns.despine(top=True, right=True, left=False, bottom=False)
plt.legend(title = None)
# plt.legend(df_melt['variable'].unique(), loc='upper left', bbox_to_anchor=(0, 1), prop={'size': 6})

# plt.legend(bbox_to_anchor=(0.01, 1), loc='upper left', prop={'size': 6})
plt.savefig('/exports/humgen/idenhond/projects/enformer/correlation/plots_paper/Plots_paper/Fig1_correlation_dnnhead/Weights.png', bbox_inches = 'tight', dpi = 300)
