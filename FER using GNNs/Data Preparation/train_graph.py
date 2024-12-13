import torch
import os
import random

from torch_geometric.loader import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool

import matplotlib.pyplot as plt

dataset_path = "E:\\GNN\\datasets\\graphs\\train"

dataset = []
for class_i in os.listdir(dataset_path):
    file_path = os.path.join(dataset_path, class_i)
    for filename in os.listdir(file_path):
        filename_path = os.path.join(file_path, filename)
        dataset.append(filename_path)
    
total = len(dataset)

dataset = random.sample(dataset, len(dataset))

train_dataset = dataset[:int(total * 0.8)]
val_dataset = dataset[int(total * 0.8):]

print(f"total dataset: {len(dataset)}")
print(f"train_dataset: {len(train_dataset)}")
print(f"val_dataset: {len(val_dataset)}")

train_list = list()
for train_ds in train_dataset:
    train_list.append(torch.load(train_ds))
    
val_list = list()
for val_ds in val_dataset:
    val_list.append(torch.load(val_ds))
    
train_loader = DataLoader(train_list, batch_size=128)
val_loader = DataLoader(val_list, batch_size=64)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = GCNConv(3, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 7)
        
    def forward(self, x, edge_index, batch=10):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        
        # 2. Readout layer
        x = global_mean_pool(x, batch) # [batch_size, hidden_channels]
        
        # 3. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x

model = GCN(hidden_channels = 64)
print(model)

# start here

model = GCN(hidden_channels=64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)  
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        correct += int((pred == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)  # Derive ratio of correct predictions.


train_accuracy = list()
val_accuracy = list()
for epoch in range(1, 171):
    train()
    train_acc = test(train_loader)
    val_acc = test(val_loader)
    train_accuracy.append(train_acc)
    val_accuracy.append(val_acc)
    print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Validation Acc: {val_acc:.4f}')
    
# Plotting
epochs = [i for i in range(1, 171)]
plt.plot(epochs, train_accuracy, label='train')
plt.plot(epochs, val_accuracy, label='test')
plt.legend()
plt.plot()