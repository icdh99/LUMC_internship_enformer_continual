import torch
import torch.nn as nn
import pytorch_lightning as pl

# Define your existing model with one fully connected layer that predicts gene expression for 5000 tracks
class GeneExpressionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1000, 5000)
        # ...

    def forward(self, x):
        x = self.fc1(x)
        # ...

    def training_step(self, batch, batch_idx):
        # ...

# Define the new head with 50 tracks
class NewHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc2 = nn.Linear(1000, 50)
        # ...

    def forward(self, x):
        x = self.fc2(x)
        # ...

# Create a new model by appending the new head to the existing model
class CombinedModel(pl.LightningModule):
    def __init__(self, gene_model):
        super().__init__()
        self.gene_model = gene_model
        self.new_head = NewHead()

    def forward(self, x):
        x = self.gene_model(x)
        x = self.new_head(x)
        return x

    def training_step(self, batch, batch_idx):
        # ...

    def configure_optimizers(self):
        # ...

# Load your existing model
gene_model = GeneExpressionModel.load_from_checkpoint("existing_model.ckpt")

# Freeze the weights of the existing model
for param in gene_model.parameters():
    param.requires_grad = False

# Create a new model with the new head and the frozen existing model
combined_model = CombinedModel(gene_model)

# Train the new head on the dataset for the 50 tracks
trainer = pl.Trainer()
trainer.fit(combined_model, datamodule)
