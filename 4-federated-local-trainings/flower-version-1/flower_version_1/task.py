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

import time

#log printer
import flower_version_1.log_printer as log_printer


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

def train(net, train_dataloader,val_dataloader, epochs, lr, fl_training_id, partition_id, server_round, device):
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss().to(device)

    #train
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train_losses = []
    val_losses = []
    val_accs = []
    log_printer.create_training_graph(fl_training_id=fl_training_id,
                                      partition_id=partition_id,
                                      server_round=server_round,
                                      optimizer="Adam",
                                      learning_rate=lr,
                                      num_epochs=int(epochs),
                                      batch_size=4)

    for epoch_idx in range(int(epochs)):
        net.train()
        epoch_running_loss = 0.0
        trained_batch = 0
        epoch_start_time = time.perf_counter()
        for batch_idx, (images, labels) in enumerate(train_dataloader): 
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = net(images)  
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            epoch_running_loss += loss.item()
            trained_batch = batch_idx + 1

        epoch_finished_time = time.perf_counter()
        epoch_elapsed_time = epoch_finished_time - epoch_start_time

        epoch_average_loss = epoch_running_loss / trained_batch
        val_loss, val_accuracy = validate(net, val_dataloader, device)
        val_losses.append(val_loss)
        val_accs.append(val_accuracy)

        current_epoch = epoch_idx + 1
        log_printer.add_one_epoch_training_graph_point(fl_training_id=fl_training_id, 
                                                       partition_id=partition_id, 
                                                       server_round=server_round,
                                                       current_epoch=current_epoch,
                                                       trained_batch=trained_batch,
                                                       train_loss=epoch_average_loss,
                                                       val_loss=val_loss,
                                                       accuracy=val_accuracy,
                                                       epoch_elapsed_time=epoch_elapsed_time)
        
    def get_avg(arr):
        return sum(arr)/len(arr)

    avg_train_loss = get_avg(train_losses) 
    avg_val_loss =get_avg(val_losses)
    avg_val_acc = get_avg(val_accs)

    return avg_train_loss, avg_val_loss, avg_val_acc

@torch.no_grad()
def validate(net,val_dataloader, device):
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
    
    val_loss = total_loss / max(total, 1)
    val_accuracy = correct / max(total, 1)

    return val_loss, val_accuracy


@torch.no_grad()
def test(net, test_dataloader, fl_training_id, partition_id, server_round,  device):
    if server_round - 1 < 1e-3:
        log_printer.create_testing_graph(fl_training_id=fl_training_id,partition_id=partition_id)
        
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

    log_printer.add_one_server_round_testing_graph_point(fl_training_id=fl_training_id, 
                                                       partition_id=partition_id, 
                                                       server_round=server_round,
                                                       criterion="CrossEntropyLoss",
                                                       batch_size=len(test_dataloader),
                                                       test_loss=test_loss,
                                                       accuracy=test_accuracy,
                                                       )
    return test_loss, test_accuracy
