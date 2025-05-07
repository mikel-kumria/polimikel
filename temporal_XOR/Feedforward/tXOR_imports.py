import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import snntorch as snn
from snntorch import surrogate
import numpy as np

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    spike_counts = []
    
    with torch.no_grad():
        for data, labels in loader:
            data = data.to(device)
            labels = labels.to(device)
            data = data.permute(1, 0, 2)
            
            # Forward pass
            outs, avg_out = model(data)
            
            # Compute loss
            loss = criterion(avg_out, labels)
            total_loss += loss.item()
            
            # Compute accuracy (get the index of maximum value)
            pred = torch.argmax(avg_out, dim=1)
            true = torch.argmax(labels, dim=1)
            correct += (pred == true).sum().item()
            total += labels.size(0)
            
            # Collect spike counts (only from the hidden layer)
            spike_counts.append(outs[-10:].sum().item() / (10 * labels.size(0)))  # Average spikes per neuron per time step
    
    accuracy = 100 * correct / total
    avg_loss = total_loss / len(loader)
    avg_spikes = np.mean(spike_counts)
    
    return avg_loss, accuracy, avg_spikes

def monitor_spike_activity(model, data, device):
    model.eval()
    with torch.no_grad():
        data = data.to(device)
        data = data.permute(1, 0, 2)
        outs, _ = model(data)
        return outs 