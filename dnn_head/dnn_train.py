import torch
import torch.nn as nn
from collections import OrderedDict
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import Dataset, DataLoader
	# dataset stores samples + labels
	# dataloader wraps iterable around dataset --> easy access

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
input_np = tensor_inputdata.numpy()
output_np = tensor_outputdata.numpy()

X_train, X_test, Y_train, Y_test = train_test_split(input_np, output_np, test_size=0.33, random_state=42)
print(f'shape of X train: {X_train.shape}')
print(f'shape of Y train: {Y_train.shape}')
print(f'shape of X test: {X_test.shape}')
print(f'shape of Y test: {Y_test.shape}')

# print(f'X train 1: \n{X_train[1].shape}\n {X_train[1]}')
# print(f'Y train 1: \n{Y_train[1].shape}\n {Y_train[1]}')

## processing data
class Data(Dataset):
	def __init__(self, X_train, y_train):
		# need to convert float64 to float32 else 
		# will get the following error
		# RuntimeError: expected scalar type Double but found Float
		self.X = torch.from_numpy(X_train.astype(np.float32))
		# need to convert float64 to Long else 
		# will get the following error
		# RuntimeError: expected scalar type Long but found Float
		self.y = torch.from_numpy(Y_train.astype(np.float32))
		self.len = self.X.shape[0]

	def __getitem__(self, index):
		return self.X[index], self.y[index]

	def __len__(self):
		return self.len

traindata = Data(X_train, Y_train)
print(f'length traindata: {len(traindata)}')
testdata = Data(X_test, Y_test)
print(f'length testdata: {len(testdata)}')

# print(type(traindata[1])) 	# tuple
# print(type(traindata[1][0]))	# tensir
# print(traindata[1])
# print(traindata[1][0])

# initializations
batch_size = 4
epochs = 2
learning_rate = 1e-2

# load training data
trainloader = DataLoader(traindata, batch_size=batch_size, shuffle = True, num_workers = 2)

## model architecture (enformer output head)
class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		self.linear = nn.Linear(in_features = 3072, out_features = 5313, bias = True)
		self.softplus = nn.Softplus(beta = 1, threshold = 20)	# default values for nn.Softplus()
	
	def forward(self, x):
		x = self.linear(x)
		x = self.softplus(x)
		return x

# initialize model
clf = Network().to(device)
print(f'model architecture: \n{clf.parameters}')

# initialize loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(clf.parameters(), lr=learning_rate)

for epoch in range(epochs):
	running_loss = 0.0
	train_acc = 0
	samples = 0
	clf.train()

	for i, data in enumerate(trainloader, 0):
		inputs, labels = data
		inputs = inputs.to(device)
		labels = labels.to(device)
		# set optimizer to zero grad to remove previous epoch gradients
		optimizer.zero_grad()
		# forward propagation
		outputs = clf(inputs)
		loss = criterion(outputs, labels)
		# backward propagation
		loss.backward()
		# optimize
		optimizer.step()
		running_loss += loss.item()

		samples += labels.size(0)
		print(f'nr of samples: {samples}')
  	# display statistics
	print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.5f}')
	print(f'[epoch {epoch + 1}, {i + 1:5d}] test loss: {running_loss / samples}')


# initialize variables for testing
test_loss = 0
test_acc = 0
samples = 0
clf.eval()
correct = 0
total = 0

# load test data
testloader = DataLoader(testdata, batch_size=batch_size, shuffle=True, num_workers=2)
with torch.no_grad():
	for i, data in enumerate(trainloader, 0):
		inputs, labels = data
		inputs = inputs.to(device)
		labels = labels.to(device)
		
		predictions = clf(inputs)

		loss = criterion(predictions, labels)

		test_loss += loss.item()

		__, predicted = torch.max(predictions.data, 1)

		total += labels.size(0)

		print(f'predicted shape: {predicted.shape}')
		print(f'labels shape: {labels.shape}')
		correct += (predicted == labels).sum().item()

		# test_acc += (predictions.max(1)[1] == labels).sum().item()

		samples += labels.size(0)
	
	print(f'[epoch {epoch + 1}, {i + 1:5d}] test loss: {test_loss / samples} test accuracy: {100 * correct // total}% \n samples: {samples}')


"""
voorbeelden van sites


# number of features (len of X cols)
input_dim = 4
# number of hidden layers
hidden_layers = 25
# number of classes (unique of y)
output_dim = 3 

class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		self.linear1 = nn.Linear(input_dim, hidden_layers)
		self.linear2 = nn.Linear(hidden_layers, output_dim)
	
	def forward(self, x):
		x = torch.sigmoid(self.linear1(x))
		x = self.linear2(x)
		return x

clf = Network()
print(clf.parameters)



## model architecture van site
def get_training_model(inFeatures = 4, hiddenDim = 8, nbClasses = 3):
	mlpModel = nn.Sequential(OrderedDict([("hidden_layer_1", 
					nn.Linear(inFeatures, hiddenDim)), 
					("activation_1", nn.ReLU()), 
					("output_layer", nn.Linear(hiddenDim, nbClasses))]))
	return mlpModel

model = get_training_model()
print(model.parameters)
"""