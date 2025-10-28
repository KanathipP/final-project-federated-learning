import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import mne
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.base import TransformerMixin, BaseEstimator

#data cleaner
import flower_version_1.motor_imaginary_train_data_cleaner as motor_imaginary_train_data_cleaner

#model
from flower_version_1.model import EEGNet

device = "cuda:0" if torch.cuda.is_available() else "cpu"

in_channel = 22
num_classes = 4 # [5, 6, 7, 8]

class Net(EEGNet):
    def __init__(self):
        super().__init__(in_channel=in_channel,num_classes=num_classes)

# data pipeline
def load_data(partition_id: int):
    train_features, train_labels, val_features, val_labels = motor_imaginary_train_data_cleaner.pipeline(partition_id)
    
    # create dataloader
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    return train_dataloader, val_dataloader


# =========================
# Train / Test
# =========================
def train(net, train_dataloader, epochs, lr, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    net.train()
    running_loss = 0.0
    for _ in range(int(epochs)):
        for images, labels in train_dataloader:  # images: (B,1,C,T)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = net(images)  # EEGNet ส่ง logits ออก
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return running_loss / max(len(train_dataloader), 1)


@torch.no_grad()
def test(net, val_dataloader, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    net.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in val_dataloader:
        images, labels = images.to(device), labels.to(device)
        logits = net(images)
        total_loss += criterion(logits, labels).item() * images.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += images.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)
