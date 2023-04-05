from datetime import datetime
start = datetime.now()
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from data_class_withval import MyDataset_train, MyDataset_val
# from model_class import model
import pytorch_lightning as pl
from pytorch_lightning.callbacks import RichProgressBar, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

# class MyDataset_train(Dataset):
#     def __init__(self, list_IDs):
#         assert type(list_IDs) == list, f'list IDs expected type list but got {type(list_IDs)}'
#         self.list_IDs = list_IDs

#     def __len__(self):
#         return len(self.list_IDs)
    
#     def __getitem__(self, index):
#         ID = self.list_IDs[index]
#         # load data and get label
#         x = torch.load(f'/exports/humgen/idenhond/data/Enformer_train/Enformer_train_embeddings_pretrainedmodel/embeddings_seq{ID}.pt', map_location=torch.device('cpu'))
#         y = torch.load(f'/exports/humgen/idenhond/data/Enformer_train/Enformer_train_targets_newtracks2703/targets_seq{ID}.pt', map_location=torch.device('cpu'))
#         y = torch.squeeze(y)
#         return x, y

# class MyDataset_val(Dataset):
#     def __init__(self, list_IDs):
#         assert type(list_IDs) == list, f'list IDs expected type list but got {type(list_IDs)}'
#         self.list_IDs = list_IDs

#     def __len__(self):
#         return len(self.list_IDs)
    
#     def __getitem__(self, index):
#         ID = self.list_IDs[index]
#         # load data and get label
#         x = torch.load(f'/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_embeddings_pretrainedmodel_perseq/embeddings_seq{ID}.pt', map_location=torch.device('cpu'))
#         y = torch.load(f'/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_targets_newtracks2703/targets_seq{ID}.pt', map_location=torch.device('cpu'))
#         y = torch.squeeze(y)
#         return x, y
    
class MyDataset_Train_NewHead(Dataset):
    def __init__(self, list_IDs):
        assert type(list_IDs) == list, f'list IDs expected type list but got {type(list_IDs)}'
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        ID = self.list_IDs[index]
        # load data and get label
        x = torch.load(f'/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_embeddings_pretrainedmodel_perseq/embeddings_seq{ID}.pt', map_location=torch.device('cpu'))
        y = torch.load(f'/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_targets_newtracks2703/targets_seq{ID}.pt', map_location=torch.device('cpu'))
        y = torch.squeeze(y)
        return x, y
    
class MyDataset_Val_NewHead(Dataset):
    def __init__(self, list_IDs):
        assert type(list_IDs) == list, f'list IDs expected type list but got {type(list_IDs)}'
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.list_IDs)
    
    def __getitem__(self, index):
        ID = self.list_IDs[index]
        # load data and get label
        x = torch.load(f'/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_embeddings_pretrainedmodel_perseq/embeddings_seq{ID}.pt', map_location=torch.device('cpu'))
        y = torch.load(f'/exports/humgen/idenhond/data/Enformer_validation/Enformer_validation_targets_newtracks2703/targets_seq{ID}.pt', map_location=torch.device('cpu'))
        y = torch.squeeze(y)
        return x, y
    
class OriginalModel(pl.LightningModule):
	def __init__(self):
		# define model
		super(OriginalModel, self).__init__()
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
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        self.new_head = NewHead()
        self.lr = 1e-4
        self.loss = nn.PoissonNLLLoss()
        self.save_hyperparameters()

    def forward(self, x):
        original_output = self.original_model(x)
        new_head_output = self.new_head(original_output)
        return original_output, new_head_output
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr = self.lr)

    def training_step(self, train_batch, batch_idx):
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
	    x, y = valid_batch
	    logits = self.forward(x)
	    val_loss = self.loss(logits, y)
	    self.log("val_loss", val_loss, prog_bar=True, sync_dist=True)
	    self.logger.experiment.add_scalars('loss', {'valid': val_loss},self.global_step)
	    return val_loss


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
    training_set = MyDataset_Train_NewHead(partition_indices['train'])
    print(f'number of train samples: {len(training_set)}')
    training_generator = DataLoader(training_set, 
                                    batch_size = BATCH_SIZE, 
                                    shuffle = True, # shuffle sample volgorde in elke epoch? zoiets
                                    num_workers = NUM_WORKERS,
                                    pin_memory = True)
    print(f'number of train batches: {len(training_generator)}')
    val_set = MyDataset_Val_NewHead(partition_indices['val'])
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
    logger = TensorBoardLogger('tb_logs', name = 'add_head_2803')
    print('tb logs folder: tb_logs/add_head_2803')
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

    original_model = OriginalModel.load_from_checkpoint('/exports/humgen/idenhond/projects/enformer/dnn_head/dnn_head_train/model_2023-03-10 17:52:03.039827_v3/epoch=19-step=5320-val_loss=0.8.ckpt')

    for param in original_model.parameters(): 
        param.requires_grad = False
    new_head = NewHead()

    combined_model = CombinedModel(original_model)

    trainer = pl.Trainer(max_epochs = EPOCHS, 
                        callbacks = callbacks, 
                        enable_checkpointing = True, 
                        logger = logger, 
                        accelerator = 'auto', 
                        num_nodes = 1,
                        devices = 2, 
                        precision = 16,
                        strategy=strategy) 
    
    trainer.fit(combined_model, training_generator, val_generator)

    print(f'Time after fit: {datetime.now() - start}\n') 

if __name__ == "__main__":
    main()