3/2/2023

This folder contains the data and the code to run a small DNN on the embeddings of the enformer model

Data: 
X = tensor_embeddingsvalidation_100.pt
Y = tensor_targetvalidation_100.pt
Prediction with model = tensor_predictionsvalidation_100.pt

Scripts:
dnn_subsetdata.py --> generate X and Y in .pt format (subset first 100 validation sequences)
dnn_train_pl.py --> pytorch lightning implementation (experiment 2B)

Data:
output.dnn_train_pl.sh.14292048 -->report of script
scripts_tryout/model_2023-03-03 15:41:24.583775 --> stored model
scripts_tryout/lightning_logs --> logs of pl


Redundant scripts:
dnn_train.py --> pytorch only implementation
dnn_train_pl_gpu.py --> pytorch lightning gpu implementation (I don't know if it is working)
dnn_lookattensors.py --> look at first tensor of X and Y