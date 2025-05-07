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

def plot_dataset_samples(train_dataset, val_dataset, results_dir, timestamp):
    """Plot 10 samples from both training and validation datasets."""
    plt.figure(figsize=(15, 10))
    
    # Plot training samples
    for i in range(10):
        plt.subplot(2, 10, i + 1)
        sample, label = train_dataset[i]
        plt.imshow(sample.T, aspect='auto', cmap='viridis')
        plt.title(f'Train {i}\nLabel: {label.argmax().item()}')
        if i == 0:
            plt.ylabel('Input\nNeuron')
        plt.xlabel('Time Step')
    
    # Plot validation samples
    for i in range(10):
        plt.subplot(2, 10, i + 11)
        sample, label = val_dataset[i]
        plt.imshow(sample.T, aspect='auto', cmap='viridis')
        plt.title(f'Val {i}\nLabel: {label.argmax().item()}')
        if i == 0:
            plt.ylabel('Input\nNeuron')
        plt.xlabel('Time Step')
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, f'dataset_samples_{timestamp}.png')
    plt.savefig(plot_path)
    plt.close()

def main():
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join(os.path.dirname(__file__), 'Results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Create timestamp for unique filenames
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Dataset parameters
    dataset_params = {
        'num_samples_train': 1000,
        'num_samples_val': 200,
        'seq_len': 30,
        'v_th': 2.0,  # Input voltage threshold
    }
    
    # Network parameters
    network_params = {
        'input_size': 3,
        'hidden_size': 100,
        'output_size': 2,
        'beta': 0.9,  # LIF neuron decay rate
        'threshold': 0.5,  # LIF neuron threshold
        'spike_grad_slope': 25,  # Surrogate gradient slope
        'weight_gain': 2.0,  # Weight initialization gain
    }
    
    # Training parameters
    training_params = {
        'num_epochs': 100,
        'batch_size': 32,
        'learning_rate': 5e-4,
        'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    }
    
    # Create datasets
    train_dataset = TemporalXORDataset(
        num_samples=dataset_params['num_samples_train'],
        v_th=dataset_params['v_th'],
        seq_len=dataset_params['seq_len']
    )
    val_dataset = TemporalXORDataset(
        num_samples=dataset_params['num_samples_val'],
        v_th=dataset_params['v_th'],
        seq_len=dataset_params['seq_len']
    )
    
    # Plot dataset samples (comment out if not needed)
    plot_dataset_samples(train_dataset, val_dataset, results_dir, timestamp)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=training_params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=training_params['batch_size'])
    
    # Initialize model
    model = TemporalXORNetwork(
        input_size=network_params['input_size'],
        hidden_size=network_params['hidden_size'],
        output_size=network_params['output_size'],
        beta=network_params['beta'],
        threshold=network_params['threshold'],
        spike_grad_slope=network_params['spike_grad_slope'],
        weight_gain=network_params['weight_gain']
    ).to(training_params['device'])
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=training_params['learning_rate'])
    
    # Training loop
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    train_spikes = []
    val_spikes = []
    
    for epoch in range(training_params['num_epochs']):
        # Training
        train_loss, train_acc, train_spike = train(
            model, train_loader, optimizer, criterion, 
            training_params['device']
        )
        
        # Validation
        val_loss, val_acc, val_spike = validate(
            model, val_loader, criterion, 
            training_params['device']
        )
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        train_spikes.append(train_spike)
        val_spikes.append(val_spike)
        
        print(f'Epoch [{epoch+1}/{training_params["num_epochs"]}]')
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
        'val_spikes': val_spikes,
        'dataset_params': dataset_params,
        'network_params': network_params,
        'training_params': training_params
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
    
    # Save model and parameters
    model_path = os.path.join(results_dir, f'model_{timestamp}.pt')
    torch.save({
        'epoch': training_params['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
        'dataset_params': dataset_params,
        'network_params': network_params,
        'training_params': training_params
    }, model_path)

if __name__ == "__main__":
    main() 