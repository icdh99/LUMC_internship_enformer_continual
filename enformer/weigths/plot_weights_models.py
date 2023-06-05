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

path = '/exports/humgen/idenhond/projects/enformer/dnn_head/dnn_head_train/model_2023-03-10 17:52:03.039827_v3/epoch=19-step=5320-val_loss=0.8.ckpt'
print(path)
model = model_dnnhead.load_from_checkpoint(path)
model.eval()
params = [param for param in model.parameters()]
print(params[0].shape)
w = params[0].detach().numpy()
w = pd.DataFrame(w)
df_reset = w.reset_index() 
df_melted = pd.melt(df_reset, id_vars='index', var_name='x', value_name='value')
df_melted['x'] = pd.to_numeric(df_melted['x'])

plt.figure()
sns.lineplot(data=df_melted, x='x', y='value', hue='index', legend=False)
plt.xlabel('X')
plt.ylabel('Value')
plt.title('Weights for DNN head model (shape 5313 x 3072)')
plt.savefig('weights_dnnhead.png')
plt.close()

plt.figure()
sns.lineplot(data=df_melted, x='x', y='value', hue='index', legend=False)
plt.xlabel('X')
plt.ylabel('Value')
plt.ylim(bottom = -10, top = 15)
plt.title('Weights for DNN head model (shape 5313 x 3072)')
plt.savefig('weights_dnnhead_scale.png')
plt.close()

class model_atac(pl.LightningModule):
	def __init__(self):
		# define model
		super(model_atac, self).__init__()
		self.linear = nn.Linear(in_features = 3072, out_features = 66, bias = True) #ADJUST NUMBER OF OUTPUT FEATURES
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

# load model trained on human atac seq tracks
path = '/exports/humgen/idenhond/projects/enformer/dnn_head/train_human_atac/model_2023-04-24 17:53:40.485828/epoch=18-step=5054-val_loss=0.4.ckpt'
print(path)
model = model_atac.load_from_checkpoint(path)
model.eval()    

params = [param for param in model.parameters()]
param_shapes = [param.shape for param in model.parameters()]
print(param_shapes)
print(params[0].shape)
w = params[0].detach().numpy()
w = pd.DataFrame(w)
# print(w)
df_reset = w.reset_index() 
df_melted = pd.melt(df_reset, id_vars='index', var_name='x', value_name='value')
df_melted['x'] = pd.to_numeric(df_melted['x'])
print(df_melted)

plt.figure()
sns.lineplot(data=df_melted, x='x', y='value', hue='index', legend=False)
plt.xlabel('X')
plt.ylabel('Value')
plt.title('Weights for DNN head ATAC model (shape 66 x 3072)')
plt.savefig('weights_dnnhead_atac.png')
plt.close()

plt.figure()
sns.lineplot(data=df_melted, x='x', y='value', hue='index', legend=False)
plt.xlabel('X')
plt.ylabel('Value')
plt.ylim(bottom = -10, top = 15)
plt.title('Weights for DNN head ATAC model (shape 66 x 3072)')
plt.savefig('weights_dnnhead_atac_scale.png')
plt.close()