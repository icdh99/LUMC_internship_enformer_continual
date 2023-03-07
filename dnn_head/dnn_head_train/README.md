This folder contains the code to train the head of the Enformer model with our own code

Data
/exports/humgen/idenhond/data/Enformer_train --> train data

Scripts
data_class.py --> subclass of Data to get x (embeddings) and y (targets) without storing all 34021 samples in memory
train.py --> code to train the model

Redundant scripts