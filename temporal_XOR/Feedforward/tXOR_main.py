import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

from tXOR_dataset import TemporalXORDataset
from tXOR_feedforward import TemporalXORNetwork
from tXOR_imports import train, validate

def plot_grid_samples(dataset, results_dir, timestamp, prefix):
    """Plot a 5x5 grid of samples from the dataset, showing only neurons 0 and 1, maximizing subplot size."""
    n_rows, n_cols = 5, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 15))
    for i in range(n_rows * n_cols):
        ax = axes[i // n_cols, i % n_cols]
        sample, label = dataset[i]
        # Optionally pad to square for perfect squares:
        data = sample.T[:2]
        if data.shape[0] < data.shape[1]:
            # Pad rows (neurons) to match columns (timesteps)
            pad = ((0, data.shape[1] - data.shape[0]), (0, 0))
        else:
            # Pad columns (timesteps) to match rows (neurons)
            pad = ((0, 0), (0, data.shape[0] - data.shape[1]))
        data_sq = np.pad(data, pad, mode='constant')
        ax.imshow(data_sq, aspect='auto', cmap='viridis', interpolation='none')
        ax.set_title(f'{prefix} {i}\nLabel: {label.argmax().item()}', fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)
    plot_path = os.path.join(results_dir, f'{prefix.lower()}_samples_{timestamp}.png')
    plt.savefig(plot_path, dpi=200, bbox_inches='tight', pad_inches=0)
    plt.close()

def plot_sample_traces(dataset, results_dir, timestamp, prefix, num_samples=5):
    """Plot line traces of neuron 0 and neuron 1 for num_samples random samples from the dataset."""
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    for idx, sample_idx in enumerate(indices):
        sample, label = dataset[sample_idx]
        timesteps = np.arange(sample.shape[0])
        plt.figure(figsize=(8, 4))
        plt.plot(timesteps, sample[:, 0], label='Neuron 0 (A)', marker='o')
        plt.plot(timesteps, sample[:, 1], label='Neuron 1 (B)', marker='o')
        plt.xlabel('Timestep')
        plt.ylabel('Input Value')
        plt.title(f'{prefix} Sample {sample_idx} | Label: {label.argmax().item()}')
        plt.legend()
        plt.tight_layout()
        plot_path = os.path.join(results_dir, f'{prefix.lower()}_trace_{sample_idx}_{timestamp}.png')
        plt.savefig(plot_path, dpi=150)
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
        'num_epochs': 1,
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
    
    # Plot sample traces (comment out if not needed)
    plot_sample_traces(train_dataset, results_dir, timestamp, prefix='Train', num_samples=5)
    plot_sample_traces(val_dataset, results_dir, timestamp, prefix='Val', num_samples=5)
    
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
    criterion = nn.BCEWithLogitsLoss()
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