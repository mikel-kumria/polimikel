import matplotlib.pyplot as plt
import numpy as np
import torch
import os

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

def plot_spike_pattern(spike_data, title="Spike Pattern"):
    """
    Plot spike patterns for a single sample
    spike_data: tensor of shape [seq_len, batch_size, output_size]
    """
    plt.figure(figsize=(12, 6))
    
    # Convert to numpy for plotting
    spike_data = spike_data.cpu().numpy()
    
    # Plot each output neuron's spikes
    for i in range(spike_data.shape[2]):
        plt.plot(spike_data[:, 0, i], label=f'Output {i}')
    
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Spike Value')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_training_metrics(train_losses, val_losses, val_accuracies, val_spikes):
    """
    Plot training metrics over epochs
    """
    plt.figure(figsize=(15, 5))
    
    # Plot losses
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot accuracy
    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)
    
    # Plot spike counts
    plt.subplot(1, 3, 3)
    plt.plot(val_spikes)
    plt.title('Average Spike Count')
    plt.xlabel('Epoch')
    plt.ylabel('Spikes')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_input_pattern(data, title="Input Pattern"):
    """
    Plot the input pattern for a single sample
    data: tensor of shape [seq_len, 3]
    """
    plt.figure(figsize=(12, 6))
    
    # Convert to numpy for plotting
    data = data.cpu().numpy()
    
    # Plot each input channel
    channels = ['A', 'B', 'R']
    for i in range(3):
        plt.plot(data[:, i], label=channels[i])
    
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Input Value')
    plt.legend()
    plt.grid(True)
    plt.show() 