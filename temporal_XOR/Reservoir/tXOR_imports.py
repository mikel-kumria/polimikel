import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

def train(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    total_spikes = 0
    
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        # Permute data to [time_steps, batch_size, input_size]
        data = data.permute(1, 0, 2)
        
        optimizer.zero_grad()
        
        # Forward pass
        spk_rec, outputs = model(data)
        
        # Calculate loss on last 10 timesteps
        loss = criterion(outputs, targets)  # outputs is already averaged over last 10 timesteps
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        pred = torch.argmax(outputs, dim=1)
        true = torch.argmax(targets, dim=1)
        correct += (pred == true).sum().item()
        total += targets.size(0)
        
        # Calculate average spikes
        total_spikes += spk_rec.sum().item()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    avg_spikes = total_spikes / (total * model.fc1.out_features * data.size(0))
    
    return avg_loss, accuracy, avg_spikes

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    total_spikes = 0
    
    with torch.no_grad():
        for data, targets in val_loader:
            data = data.to(device)
            targets = targets.to(device)
            # Permute data to [time_steps, batch_size, input_size]
            data = data.permute(1, 0, 2)
            
            # Forward pass
            spk_rec, outputs = model(data)
            
            # Calculate loss
            loss = criterion(outputs, targets)  # outputs is already averaged over last 10 timesteps
            
            # Calculate accuracy
            pred = torch.argmax(outputs, dim=1)
            true = torch.argmax(targets, dim=1)
            correct += (pred == true).sum().item()
            total += targets.size(0)
            
            # Calculate average spikes
            total_spikes += spk_rec.sum().item()
            total_loss += loss.item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total
    avg_spikes = total_spikes / (total * model.fc1.out_features * data.size(0))
    
    return avg_loss, accuracy, avg_spikes 