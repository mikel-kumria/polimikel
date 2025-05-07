import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from .tXOR_dataset import TemporalXORDataset
from .tXOR_feedforward import TemporalXORNetwork
from .tXOR_imports import train, validate

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Hyperparameters
    num_epochs = 20
    batch_size = 32
    learning_rate = 0.001
    hidden_size = 100
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create datasets
    train_dataset = TemporalXORDataset(num_samples=1000)
    val_dataset = TemporalXORDataset(num_samples=200)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    model = TemporalXORNetwork(hidden_size=hidden_size).to(device)
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_spikes = []
    val_spikes = []
    
    for epoch in range(num_epochs):
        # Training
        train_loss, train_acc, train_spike = train(model, train_loader, optimizer, criterion, device)
        
        # Validation
        val_loss, val_acc, val_spike = validate(model, val_loader, criterion, device)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_spikes.append(train_spike)
        val_spikes.append(val_spike)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Avg Spikes: {train_spike:.2f}')
        print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Avg Spikes: {val_spike:.2f}')
        print('-' * 50)
    
    # Save training metrics
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'train_spikes': train_spikes,
        'val_spikes': val_spikes
    }
    metrics_path = os.path.join(results_dir, f'metrics_{timestamp}.npz')
    np.savez(metrics_path, **metrics)
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 3, 2)
    plt.plot(train_accs, label='Train')
    plt.plot(val_accs, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    # Plot spike counts
    plt.subplot(1, 3, 3)
    plt.plot(train_spikes, label='Train')
    plt.plot(val_spikes, label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Average Spikes')
    plt.title('Average Spike Count')
    plt.legend()
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'training_results_{timestamp}.png')
    plt.savefig(plot_path)
    plt.close()
    
    # Save model
    model_path = os.path.join(results_dir, f'model_{timestamp}.pt')
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
    }, model_path)

if __name__ == "__main__":
    main() 