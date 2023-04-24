import torch
import torch.nn as nn
import pytorch_lightning as pl

class model(pl.LightningModule):
	def __init__(self):
		# define model
		super(model, self).__init__()
		self.linear = nn.Linear(in_features = 3072, out_features = 66, bias = True)
		self.softplus = nn.Softplus(beta = 1, threshold = 20)	# default values for nn.Softplus()
		self.lr = 1e-4
		self.loss = nn.PoissonNLLLoss()
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
		self.log("train_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
		self.logger.experiment.add_scalars('loss', {'train': loss},self.global_step)
		return loss
	
	def test_step(self, batch, batch_idx):
		x, y = batch
		logits = self.forward(x)
		test_loss = self.loss(logits, y)
		self.log("test_loss", test_loss, on_epoch=True, prog_bar=True, sync_dist=True)
		return test_loss

	def validation_step(self, valid_batch, batch_idx):
		# define validation loop steps
		x, y = valid_batch
		logits = self.forward(x)
		val_loss = self.loss(logits, y)
		self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
		self.logger.experiment.add_scalars('loss', {'valid': val_loss},self.global_step)
		return val_loss

	def predict_step(self, batch, batch_idx):
		x, y = batch
		return self(x), y
	
	# def validation_epoch_end(self, out):
	# 	gathered = self.all_gather(out)
	# 	if self.global_rank == 0:
	# 		print(0)
	# 	# calculate correlation for validaton sequences at the end of each epoch
	# 	self.logger.experiment(add_scalars('correlation'), self.global_step)
	# 	return None