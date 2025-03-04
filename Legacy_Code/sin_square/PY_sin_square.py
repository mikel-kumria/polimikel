import torch
from torch import nn
import os
import json
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
from datetime import datetime
from torch.utils.data import DataLoader, Dataset
import snntorch as snn
from snntorch import spikeplot as splt
import tqdm

# Device agnostic code
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")

# Create "simulations" folder in the current directory
simulations_folder = os.path.join(os.getcwd(), 'simulations')
os.makedirs(simulations_folder, exist_ok=True)

# Create unique folder name inside the "simulations" folder
now = datetime.now()
unique_folder_name = os.path.join(simulations_folder, now.strftime("PY_sin_square_%Yy%mm%dd_%Hh%Mm"))
os.makedirs(unique_folder_name, exist_ok=True)

# Seeds for reproducibility
seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if device == 'cuda':
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)

# Parameters
params = {
    "std_dev": 0.4,
    "beta": 0.9,
    "num_samples": 1000,
    "train_ratio": 0.7,
    "test_ratio": 0.1,
    "validation_ratio": 0.2,
    "freq_min": 2,
    "freq_max": 20,
    "amp_min": 0.1,
    "amp_max": 5,
    "duration": 1,
    "input_size": 1,
    "hidden_size": 97,
    "output_size": 2,
    "batch_size": 32,
    "num_epochs": 20,
    "learning_rate": 0.001,
    "optimizer_betas": (0.99, 0.999)
}

# Dynamically compute sample_rate and deltaT based on freq_max
params["sample_rate"] = 10 * params["freq_max"]
params["deltaT"] = 1 / params["sample_rate"]
params["num_steps"] = int(params["duration"] / params["deltaT"])
params["tau"] = -params["deltaT"] / np.log(params["beta"])

with open(os.path.join(unique_folder_name, 'hyperparameters.json'), 'w') as f:
    json.dump(params, f)

# Dataset and DataLoader definitions
def generate_sinusoidal_wave(frequency, sample_rate, duration, amplitude, std_dev, phase):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    gaussian_noise = np.random.normal(0, std_dev * amplitude, wave.shape)
    return wave + gaussian_noise

def generate_square_wave(frequency, sample_rate, duration, amplitude, std_dev, phase):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sign(np.sin(2 * np.pi * frequency * t + phase))
    gaussian_noise = np.random.normal(0, std_dev * amplitude, wave.shape)
    return wave + gaussian_noise

class MixedWaveDataset(Dataset):
    def __init__(self, num_samples, sample_rate, duration, freq_min, freq_max, amp_min, amp_max, std_dev, phase):
        self.samples = []
        for _ in range(num_samples):
            frequency = np.random.uniform(freq_min, freq_max)
            amplitude = np.random.uniform(amp_min, amp_max)
            wave_type = np.random.choice(['sine', 'square'])
            phase = np.random.uniform(0, 2 * np.pi)
            wave = generate_sinusoidal_wave(frequency, sample_rate, duration, amplitude, std_dev, phase) if wave_type == 'sine' else generate_square_wave(frequency, sample_rate, duration, amplitude, std_dev, phase)
            label = 0 if wave_type == 'sine' else 1
            self.samples.append((wave, label))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        wave, label = self.samples[idx]
        return torch.tensor(wave, dtype=torch.float), label

def split_dataset(dataset, train_ratio, validation_ratio, test_ratio):
    train_size = int(len(dataset) * train_ratio)
    validation_size = int(len(dataset) * validation_ratio)
    test_size = len(dataset) - train_size - validation_size
    return torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])

def get_dataloaders(train_dataset, validation_dataset, test_dataset, batch_size):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader, test_loader

# Spiking Neural Network definition
class WaveNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size): 
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=np.random.uniform(0.8, 0.9, hidden_size), learn_beta=True, learn_threshold=True)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=np.random.uniform(0.8, 0.9, output_size), learn_beta=True, learn_threshold=True)
        
    def forward(self, x, mem1=None, mem2=None):
        if mem1 is None:
            mem1 = self.lif1.init_leaky()
        if mem2 is None:
            mem2 = self.lif2.init_leaky()

        spk1_rec, mem1_rec, spk2_rec, mem2_rec = [], [], [], []

        for step in range(params["num_steps"]):
            cur1 = self.fc1(x[:, step].unsqueeze(1))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk1_rec.append(spk1)
            mem1_rec.append(mem1)
            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk1_rec, dim=0), torch.stack(mem1_rec, dim=0), torch.stack(spk2_rec, dim=0), torch.stack(mem2_rec, dim=0)

# Training, Testing, and Validation functions
def train_model(model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for waves, labels in train_loader:
        waves, labels = waves.to(device).float(), labels.to(device).long()
        optimizer.zero_grad()
        spk1_rec, mem1_rec, spk2_rec, mem2_rec = model(waves)
        loss = criterion(spk2_rec.sum(dim=0), labels)
        loss.backward()
        optimizer.step()
        model.lif1.beta.data.clamp_(0.01, 0.99)
        model.lif2.beta.data.clamp_(0.01, 0.99)
        running_loss += loss.item()
    return running_loss / len(train_loader)

def test_model(model, test_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for waves, labels in test_loader:
            waves, labels = waves.to(device).float(), labels.to(device).long()
            spk1_rec, mem1_rec, spk2_rec, mem2_rec = model(waves)
            predicted = spk2_rec.sum(dim=0).argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def validate_model(model, validation_loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for waves, labels in validation_loader:
            waves, labels = waves.to(device).float(), labels.to(device).long()
            spk1_rec, mem1_rec, spk2_rec, mem2_rec = model(waves)
            predicted = spk2_rec.sum(dim=0).argmax(dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

def plot_wave(data_loader, num_plots=6, sample_rate=100, deltaT=0.001, save_path=None):
    data_iter = iter(data_loader)
    waves, labels = next(data_iter)
    plt.figure(figsize=(18, 6))
    for i in range(num_plots):
        plt.subplot(2, 3, i + 1)
        color = 'b' if labels[i] == 0 else 'r'
        plt.plot(waves[i].numpy(), label=f"{'Sinusoidal' if labels[i] == 0 else 'Square'} Wave", color=color)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
    if save_path:
        plt.savefig(save_path)

def plot_membrane_potential(mem1_rec, mem2_rec, hidden_neuron_idx, output_neuron_idx, num_steps, save_path=None):
    time_steps = np.arange(num_steps)
    hidden_mem_potential = mem1_rec[:, 0, hidden_neuron_idx].detach().cpu().numpy()
    output_mem_potential = mem2_rec[:, 0, output_neuron_idx].detach().cpu().numpy()
    plt.figure(figsize=(14, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_steps, hidden_mem_potential, label=f'Hidden Layer Neuron {hidden_neuron_idx}', color='blue')
    plt.xlabel('Time Steps')
    plt.ylabel('Membrane Potential (Hidden Layer)')
    plt.title(f'Membrane Potential of Hidden Layer Neuron {hidden_neuron_idx} over Time')
    plt.legend()
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(time_steps, output_mem_potential, label=f'Output Layer Neuron {output_neuron_idx}', color='orange')
    plt.xlabel('Time Steps')
    plt.ylabel('Membrane Potential (Output Layer)')
    plt.title(f'Membrane Potential of Output Layer Neuron {output_neuron_idx} over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600)

def plot_accuracies(num_epochs, validation_accuracies, test_accuracies, save_path=None):
    sns.set_style('whitegrid')
    plt.figure(figsize=(14, 6))
    cmap = plt.get_cmap('bwr')(np.linspace(0.1, 0.9, 2))
    plt.plot(range(num_epochs), test_accuracies, label='Test Accuracy', color=cmap[1])
    plt.plot(range(num_epochs), validation_accuracies, label='Validation Accuracy', color=cmap[0])
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation and Test Accuracy over Epochs')
    plt.ylim(1, 100)
    plt.xlim(0, num_epochs)
    plt.legend(shadow=True)
    sns.despine()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=600)

def plot_beta_values(beta_values, num_epochs, layer_name, save_path=None):
    beta_values = np.array(beta_values)
    plt.figure(figsize=(14, 6))
    for i in range(beta_values.shape[1]):
        plt.plot(range(num_epochs), beta_values[:, i], label=f'Neuron {i+1}')
    plt.xlabel('Epochs')
    plt.ylabel(r'$\beta$ values')
    plt.title(f'$\beta$ values ({layer_name}) over Epochs')
    plt.ylim(0.001, 1)
    plt.xlim(0, num_epochs)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=600)

def plot_tau_values(beta_values, num_epochs, layer_name, deltaT, save_path=None):
    tau_values = -1000 * deltaT / np.log(np.array(beta_values))
    plt.figure(figsize=(14, 6))
    for i in range(tau_values.shape[1]):
        plt.plot(range(num_epochs), tau_values[:, i], label=f'Neuron {i+1}')
    plt.xlabel('Epochs')
    plt.ylabel(r'$\tau$ values [ms]')
    plt.title(f'$\tau$ values ({layer_name}) over Epochs')
    plt.ylim(0.001, None)
    plt.xlim(0, num_epochs)
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=600)

def plot_weights(weights, num_epochs, layer_name, save_path=None):
    weights = np.array(weights)
    plt.figure(figsize=(14, 6))
    num_neurons = weights.shape[2]
    for i in range(num_neurons):
        plt.plot(range(num_epochs), weights[:, :, i].flatten(), label=f'Neuron {i+1}')
    plt.xlabel('Epochs')
    plt.ylabel(f'{layer_name} Weights')
    plt.title(f'{layer_name} Weights over Epochs')
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=600)

def plot_threshold(thresholds, num_epochs, layer_name, save_path=None):
    thresholds = np.array(thresholds)
    plt.figure(figsize=(14, 6))
    for i in range(thresholds.shape[1]):
        plt.plot(range(num_epochs), thresholds[:, i], label=f'Neuron {i+1}')
    plt.xlabel('Epochs')
    plt.ylabel(f'Membrane Threshold Potential ({layer_name})')
    plt.title(f'Membrane Threshold Potentials ({layer_name}) over Epochs')
    plt.legend()
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=600)

def plot_histogram(data, xlabel, ylabel, title, save_path=None):
    plt.hist(data, alpha=0.8)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    if save_path:
        plt.savefig(save_path, dpi=600)

def plot_snn_spikes(spk_in, spk1_rec, spk2_rec, title, save_path=None):
    fig, axs = plt.subplots(3, figsize=(12, 8), sharex=True)
    splt.raster(spk_in, ax=axs[0], color='tab:blue')
    axs[0].set_title('Input Layer Spikes')
    splt.raster(spk1_rec, ax=axs[1], color='tab:orange')
    axs[1].set_title('Hidden Layer Spikes')
    splt.raster(spk2_rec, ax=axs[2], color='tab:green')
    axs[2].set_title('Output Layer Spikes')
    plt.suptitle(title)
    if save_path:
        plt.savefig(save_path)

def main():
    dataset = MixedWaveDataset(
        num_samples=params["num_samples"],
        sample_rate=params["sample_rate"],
        duration=params["duration"],
        freq_min=params["freq_min"],
        freq_max=params["freq_max"],
        amp_min=params["amp_min"],
        amp_max=params["amp_max"],
        std_dev=params["std_dev"],
        phase=0
    )

    train_dataset, validation_dataset, test_dataset = split_dataset(
        dataset, 
        train_ratio=params["train_ratio"], 
        validation_ratio=params["validation_ratio"], 
        test_ratio=params["test_ratio"]
    )

    train_loader, validation_loader, test_loader = get_dataloaders(
        train_dataset, 
        validation_dataset, 
        test_dataset, 
        batch_size=params["batch_size"]
    )

    model = WaveNeuralNetwork(
        input_size=params["input_size"], 
        hidden_size=params["hidden_size"], 
        output_size=params["output_size"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(
        model.parameters(), 
        lr=params["learning_rate"], 
        betas=params["optimizer_betas"]
    )

    validation_accuracies, test_accuracies = [], []
    beta1_values, beta2_values = [], []
    weights_hidden_layer, weights_output_layer = [], []
    threshold_hidden_layer, threshold_output_layer = [], []

    pbar = tqdm.trange(params["num_epochs"], desc='Epochs')
    for epoch in pbar:
        train_loss = train_model(model, train_loader, criterion, optimizer)
        validation_accuracy = validate_model(model, validation_loader)
        test_accuracy = test_model(model, test_loader)
        validation_accuracies.append(validation_accuracy)
        test_accuracies.append(test_accuracy)
        beta1_values.append(model.lif1.beta.detach().cpu().numpy().flatten())
        beta2_values.append(model.lif2.beta.detach().cpu().numpy().flatten())
        threshold_hidden_layer.append(model.lif1.threshold.detach().cpu().numpy().flatten())
        threshold_output_layer.append(model.lif2.threshold.detach().cpu().numpy().flatten())
        weights_hidden_layer.append(model.fc1.weight.detach().cpu().numpy().copy())
        weights_output_layer.append(model.fc2.weight.detach().cpu().numpy().copy())
        if epoch == 0 or epoch == params["num_epochs"] - 1 or epoch % 5 == 0:
            print(f'Epoch [{epoch + 1}/{params["num_epochs"]}]  ||  Test Accuracy: {test_accuracy:.2f}%  ||  Validation Accuracy: {validation_accuracy:.2f}%  ||  Loss: {train_loss:.4f}')
        pbar.set_postfix({'Test Accuracy': test_accuracy, 'Validation Accuracy': validation_accuracy, 'Loss': train_loss})

    torch.save(model.state_dict(), os.path.join(unique_folder_name, 'model.pth'))

    # Plot wave samples
    plot_wave(train_loader, save_path=os.path.join(unique_folder_name, 'wave_samples.png'))

    # Plot accuracy
    plot_accuracies(params["num_epochs"], validation_accuracies, test_accuracies, save_path=os.path.join(unique_folder_name, 'accuracy_plot.png'))

    # Plot beta values
    plot_beta_values(beta1_values, params["num_epochs"], 'hidden layer', save_path=os.path.join(unique_folder_name, 'beta_hidden_layer.png'))
    plot_beta_values(beta2_values, params["num_epochs"], 'output layer', save_path=os.path.join(unique_folder_name, 'beta_output_layer.png'))

    # Plot tau values
    plot_tau_values(beta1_values, params["num_epochs"], 'hidden layer', params["deltaT"], save_path=os.path.join(unique_folder_name, 'tau_hidden_layer.png'))
    plot_tau_values(beta2_values, params["num_epochs"], 'output layer', params["deltaT"], save_path=os.path.join(unique_folder_name, 'tau_output_layer.png'))

    # Plot weights
    plot_weights(weights_hidden_layer, params["num_epochs"], 'Hidden Layer', save_path=os.path.join(unique_folder_name, 'weights_hidden_layer.png'))
    plot_weights(weights_output_layer, params["num_epochs"], 'Output Layer', save_path=os.path.join(unique_folder_name, 'weights_output_layer.png'))

    # Plot threshold potentials
    plot_threshold(threshold_hidden_layer, params["num_epochs"], 'Hidden Layer', save_path=os.path.join(unique_folder_name, 'threshold_hidden_layer.png'))
    plot_threshold(threshold_output_layer, params["num_epochs"], 'Output Layer', save_path=os.path.join(unique_folder_name, 'threshold_output_layer.png'))

    # Histogram of beta values
    plot_histogram(model.lif1.beta.detach().cpu().numpy(), r'$\beta$ values', 'Frequency', r'Histogram of $\beta$ values for hidden layer neurons', save_path=os.path.join(unique_folder_name, 'beta_histogram_hidden_layer.png'))
    plot_histogram(model.lif2.beta.detach().cpu().numpy(), r'$\beta$ values', 'Frequency', r'Histogram of $\beta$ values for output layer neurons', save_path=os.path.join(unique_folder_name, 'beta_histogram_output_layer.png'))

    # Histogram of tau values
    plot_histogram(-1000 * params["deltaT"] / np.log(model.lif1.beta.detach().cpu().numpy()), r'$\tau$ (ms)', 'Frequency', r'Histogram of $\tau$ values for hidden layer neurons', save_path=os.path.join(unique_folder_name, 'tau_histogram_hidden_layer.png'))
    plot_histogram(-1000 * params["deltaT"] / np.log(model.lif2.beta.detach().cpu().numpy()), r'$\tau$ (ms)', 'Frequency', r'Histogram of $\tau$ values for output layer neurons', save_path=os.path.join(unique_folder_name, 'tau_histogram_output_layer.png'))

    # Plot spiking activity
    waves, labels = next(iter(test_loader))
    waves, labels = waves.to(device).float(), labels.to(device).long()
    spk1_rec, mem1_rec, spk2_rec, mem2_rec = model(waves)
    spk_in = waves
    plot_snn_spikes(spk_in, spk1_rec, spk2_rec, "Fully Connected Spiking Neural Network", save_path=os.path.join(unique_folder_name, 'snn_spikes.png'))

    # Plot membrane potential
    hidden_neuron_idx = 0
    output_neuron_idx = 0
    plot_membrane_potential(mem1_rec, mem2_rec, hidden_neuron_idx, output_neuron_idx, params["num_steps"], save_path=os.path.join(unique_folder_name, 'membrane_potential.png'))

if __name__ == "__main__":
    main()