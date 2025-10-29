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
import flower_version_1.motor_imaginary_data_cleaner as motor_imaginary_data_cleaner

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
    train_features, train_labels, val_features, val_labels, test_features, test_labels = motor_imaginary_data_cleaner.pipeline(partition_id)
    
    # create dataloader
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader

def train(net, train_dataloader,val_dataloader, epochs, lr, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    #train
    net.train()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    running_loss = 0.0
    for _ in range(int(epochs)):
        for images, labels in train_dataloader: 
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = net(images)  
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    
    train_loss = running_loss / max(len(train_dataloader), 1)
    
    # validation
    net.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in val_dataloader:
        images, labels = images.to(device), labels.to(device)
        logits = net(images)
        total_loss += criterion(logits, labels).item() * images.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += images.size(0)
    
    val_loss = total_loss / max(total, 1)
    val_accuracy = correct / max(total, 1)

    return train_loss, val_loss, val_accuracy


@torch.no_grad()
def test(net, test_dataloader, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    net.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in test_dataloader:
        images, labels = images.to(device), labels.to(device)
        logits = net(images)
        total_loss += criterion(logits, labels).item() * images.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += images.size(0)
    test_loss = total_loss / max(total, 1)
    test_accuracy = correct / max(total, 1)
    return test_loss, test_accuracy
