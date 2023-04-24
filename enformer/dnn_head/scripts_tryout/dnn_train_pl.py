from datetime import datetime
start = datetime.now()
import torch
import torch.nn as nn
import pytorch_lightning as pl
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from datetime import datetime
import os

# setting device on GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device (set to GPU if available):', device)

## load subset of input data X (embeddings)
path_inputdata = 'tensor_embeddingsvalidation_100.pt'
tensor_inputdata = torch.load(path_inputdata)
print(f'shape of tensor with input data: {tensor_inputdata.shape}')
print(f'type of tensor with input data: {type(tensor_inputdata)}')
print(f'device of tensor with input data: {tensor_inputdata.device}')

## load output data Y (tensor flow records)
path_outputdata = 'tensor_targetvalidation_100.pt'
tensor_outputdata = torch.load(path_outputdata)
print(f'shape of tensor with output data: {tensor_outputdata.shape}')
print(f'type of tensor with output data: {type(tensor_outputdata)}')
print(f'device of tensor with output data: {tensor_outputdata.device}')

## train test split for X and Y
X_trainval, X_test, Y_trainval, Y_test = train_test_split(tensor_inputdata.numpy(), tensor_outputdata.numpy(), test_size = 0.20, random_state = 42)
X_train, X_val, Y_train, Y_val = train_test_split(X_trainval, Y_trainval, test_size = 0.25, random_state = 42)
print(f'shape of X train: {X_train.shape}')
print(f'shape of Y train: {Y_train.shape}')
print(f'shape of X val: {X_val.shape}')
print(f'shape of Y val: {Y_val.shape}')
print(f'shape of X test: {X_test.shape}')
print(f'shape of Y test: {Y_test.shape}\n')

print(f'first entry of Y test:')
print(Y_test[0])
print(Y_test[0].shape)
print('\n')

print(f'Time after loading data: {datetime.now() - start}\n') 

class Data(Dataset):
	def __init__(self, X, Y):
		self.X = torch.from_numpy(X.astype(np.float32))
		self.y = torch.from_numpy(Y.astype(np.float32))
		self.len = self.X.shape[0]

	def __getitem__(self, index):
		return self.X[index], self.y[index]

	def __len__(self):
		return self.len

traindata = Data(X_train, Y_train)
print(f'number of train samples: {len(traindata)}')
valdata = Data(X_val, Y_val)
print(f'number of val samples: {len(valdata)}')
testdata = Data(X_test, Y_test)
print(f'number of test samples: {len(testdata)}\n')

class model(pl.LightningModule):
	def __init__(self):
		# define model
		super(model, self).__init__()
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
	


BATCH_SIZE = 20
EPOCHS = 1000

trainloader = DataLoader(traindata, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0)
valloader = DataLoader(valdata, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
testloader = DataLoader(testdata, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)

# print(f'length of trainloader: {len(trainloader)}')
# print(f'length of valloader: {len(valloader)}')
# # print(f'trainloader 0 : {(trainloader[0].shape)}')
# # batch = iter(trainloader)
# images, labels = next(iter(trainloader))
# print(f'shape of batch.next: {images.shape}')
# print(f'shape of labels: {labels.shape}')

# print(f'other method')
# for test_images, test_labels in trainloader:  
#     sample_image = test_images[0]    # Reshape them according to your needs.
#     print(sample_image.shape)
#     sample_label = test_labels[0]
#     print(sample_label.shape)

# tensorboard logger
logger = TensorBoardLogger('tb_logs', name = 'val100_epochs_100')

## make folder to store model in
ts = datetime.timestamp(datetime.now())
date_time = datetime.fromtimestamp(ts)
if not os.path.exists(f'./model_{date_time}/'):
	os.makedirs(f'./model_{date_time}/')
print(f'folder where model is stored: ./model_{date_time}')

# define callbacks 
early_stop_callback = EarlyStopping(monitor="val_loss", 
                                            min_delta=0.00, 
                                            patience=5, verbose=False, mode="min")

modelcheckpoint = ModelCheckpoint(monitor = 'val_loss', 
								mode = 'min', 
								save_top_k = 1, 
								dirpath = f'./model_{date_time}/', 
								filename = "{epoch}-{step}-{val_loss:.1f}")
# save_weights_only=False --> True (maar .ckpt bestand is nu nog niet zo groot dus hoeft miss nog niet, nog niet getest)

callbacks = [RichProgressBar(), early_stop_callback, modelcheckpoint]

clf = model()
trainer = pl.Trainer(max_epochs = EPOCHS, callbacks = callbacks, enable_checkpointing = True, logger = logger, accelerator = 'gpu', devices = 1) # 
trainer.fit(clf, trainloader, valloader)

print(f'Time after fit: {datetime.now() - start}\n') 

exit()
trainer.test(clf, testloader)

print(f'Time after test: {datetime.now() - start}\n') 

print(f'predictions with clf model')  # two predictions are similar
predictions = trainer.predict(clf, testloader)
print(f'predictions: {predictions}')
print('\n')

print(f'predictions with loaded model')	# two predictions are similar
folder = os.listdir(f'./model_{date_time}/')
print(f'files in folder: {folder}')
file = folder[0]
print(f'folder: {folder}')
print(f'file: {file}')
path = os.path.join(f'./model_{date_time}/', file)
print(f'path: {path}')
model = model.load_from_checkpoint(path)
model.eval()
predictions = model(torch.from_numpy(X_test.astype(np.float32)))
print(predictions)

torch.save(predictions, 'tensor_predictionsvalidation_100_goodparams.pt')
print(f'predictions are stored in tensor_predictionsvalidation_100_goodparams.pt')

# y_pred = []
# y_true = []

# for j in range(len(predictions)):

# 	y_pred.extend(predictions[j][0].detach().numpy())
# 	y_true.extend(predictions[j][1].detach().numpy())
    
# y_pred = np.array(y_pred)
# y_true = np.array(y_true)

# print(f'\ny pred shape: {y_pred.shape}')
# print(f'y true shape: {y_true.shape}')

# print(f'\ntype predictions: {type(predictions)}')
# print(f'length predictions: {len(predictions)}')	# list has length 1

# print(f'\ntype predictions[0]: {type(predictions[0])}')
# print(f'length predictions[0]: {len(predictions[0])}')

# print(f'\ntype predictions[0][0]: {type(predictions[0][0])}')
# print(f'length predictions[0][0]: {len(predictions[0][0])}')
# print(f'shape predictions[0][0]: {(predictions[0][0].shape)}')

# print(f'\ntype predictions[0][1]: {type(predictions[0][1])}')
# print(f'length predictions[0][1]: {len(predictions[0][1])}')
# print(f'shape predictions[0][1]: {(predictions[0][1].shape)}')

# print(f'\ntype predictions[0][0][0]: {type(predictions[0][0][0])}')
# print(f'length predictions[0][0][0]: {len(predictions[0][0][0])}')
# print(f'shape predictions[0][0][0]: {(predictions[0][0][0].shape)}')

print(f'Time at end of script: {datetime.now() - start}') 