from datetime import datetime
start = datetime.now()
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from data_class_withval import MyDataset_train, MyDataset_val
from model_class import model
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import os
import random
import multiprocessing as mp

def main():
    # mp.set_start_method('spawn', force=True)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print('Using device (set to GPU if available):', device)

    BATCH_SIZE = 64
    EPOCHS = 20
    NUM_WORKERS = 4
    strategy = 'ddp_find_unused_parameters_false'

    print(f'parameters: \n batch size: {BATCH_SIZE} \n max epochs: {EPOCHS} \n strategy: {strategy} \n num workers: {NUM_WORKERS}\n')

    ### make partition dictionary with IDs for train and validation
    # TODO: add validation data
    partition_indices = {}
    partition_indices['train'] = list(range(1, 34021+1))
    partition_indices['val'] = list(range(1, 2213+1))

    # generators
    training_set = MyDataset_train(partition_indices['train'])
    print(f'number of train samples: {len(training_set)}')
    training_generator = DataLoader(training_set, 
                                    batch_size = BATCH_SIZE, 
                                    shuffle = True, # shuffle sample volgorde in elke epoch? zoiets
                                    num_workers = NUM_WORKERS,
                                    pin_memory = True)
    print(f'number of train batches: {len(training_generator)}')
    val_set = MyDataset_val(partition_indices['val'])
    print(f'number of val samples: {len(val_set)}')
    val_generator = DataLoader(val_set, 
                                    batch_size = BATCH_SIZE, 
                                    shuffle = False, # shuffle sample volgorde in elke epoch? zoiets
                                    num_workers = NUM_WORKERS,
                                    pin_memory = True)
    print(f'number of validation batches: {len(val_generator)}')

    # make folder to store model in
    ts = datetime.timestamp(datetime.now())
    date_time = datetime.fromtimestamp(ts)
    if not os.path.exists(f'./model_{date_time}/'): os.makedirs(f'./model_{date_time}/')
    print(f'folder where model is stored: ./model_{date_time}')

    # tensorboard logger
    logger = TensorBoardLogger('tb_logs', name = 'dnnhead_newtracks_2703')
    print('tb logs folder: tb_logs/dnnhead_newtracks_2703')
    print(f'logger version: {logger.version}\n')

    # define callbacks 
    early_stop_callback = EarlyStopping(monitor="val_loss", 
                                                min_delta=0.001, 
                                                patience=2, verbose=True, mode="min")

    modelcheckpoint = ModelCheckpoint(monitor = 'val_loss', 
                                    mode = 'min', 
                                    save_top_k = 1, 
                                    dirpath = f'./model_{date_time}/', 
                                    filename = "{epoch}-{step}-{val_loss:.1f}")

    callbacks = [RichProgressBar(), early_stop_callback, modelcheckpoint]

    clf = model()

    # load existing model

    # freeze weigths of existing model

    # create a combined model

    # trainer + fit combined model

    trainer = pl.Trainer(max_epochs = EPOCHS, 
                        callbacks = callbacks, 
                        enable_checkpointing = True, 
                        logger = logger, 
                        accelerator = 'auto', 
                        num_nodes = 1,
                        devices = 2, 
                        precision = 16,
                        strategy=strategy) 
    trainer.fit(clf, training_generator, val_generator)

    # save model for inference
    # torch.save(model.state_dict(), 'models')

    #Later to restore:
    # model.load_state_dict(torch.load(filepath))
    # model.eval()

    print(f'Time after fit: {datetime.now() - start}\n') 

if __name__ == "__main__":
    main()


class ExistingModel(pl.LightningModule):
	def __init__(self):
		# define model
		super(ExistingModel, self).__init__()
		self.linear = nn.Linear(in_features = 3072, out_features = 5313, bias = True)
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

class NewHead(nn.Module):
	def __init__(self):
		super().__init__()
		self.linear = nn.Linear(in_features = 3072, out_features = 19)
		self.softplus = nn.Softplus(beta = 1, threshold = 20)	# default values for nn.Softplus()

	def forward(self, x):
		x = self.linear(x)
		x = self.softplus(x)
		return x

class CombinedModel(pl.LightningModule):
	def __init__(self, gene_model):
    	super().__init__()
        self.gene_model = gene_model
        self.new_head = NewHead()