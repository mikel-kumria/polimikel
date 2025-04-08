import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import snntorch as snn
import matplotlib.pyplot as plt
import random

def set_seed(seed = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.use_deterministic_algorithms(True)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_unique_folder(base_dir):
    """
    Creates a unique folder inside base_dir with the naming pattern:
      trial_N_date_yyyy_mm_dd_hh_mm,
    where N is the highest number found among folders starting with "trial_"
    plus one.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    existing = [d for d in os.listdir(base_dir)
                if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("trial_")]
    trial_numbers = []
    for folder in existing:
        match = re.match(r"trial_(\d+)_date_", folder)
        if match:
            trial_numbers.append(int(match.group(1)))
    next_trial = max(trial_numbers) + 1 if trial_numbers else 1
    now_str = datetime.now().strftime("%Y_%m_%d_%H_%M")
    folder_name = f"trial_{next_trial}_date_{now_str}"
    unique_folder = os.path.join(base_dir, folder_name)
    os.makedirs(unique_folder, exist_ok=True)
    return unique_folder

def load_tsv_input(tsv_path, sample_index=0, output_folder="."):
    """
    Loads a TSV file with the following format:
      - The first column is the label (-1 for abnormal, 1 for normal)
      - The remaining columns are numeric features (each sample is a time series)
    
    Reads the file using pandas, extracts the time series from the given sample_index,
    and saves a high-quality PNG plot of the input time series in output_folder.
    
    Returns:
        A tensor of shape (1, time_steps, 1) representing the input.
    """
    data = pd.read_csv(tsv_path, sep='\t', header=0)
    raw_labels = data.iloc[:, 0].values.astype(int)
    _ = ((raw_labels == 1).astype(int))
    features = data.iloc[:, 1:].values.astype(np.float32)
    x_sample = features[sample_index]  # shape: (time_steps,)
    
    plt.figure(figsize=(8, 4))
    plt.plot(x_sample, marker='o', linestyle='-')
    plt.title('TSV Input Time Series')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.grid(True)
    plt.tight_layout()
    input_png_path = os.path.join(output_folder, 'tsv_input.png')
    plt.savefig(input_png_path, dpi=600)
    plt.close()
    
    x_tensor = torch.tensor(x_sample, dtype=torch.float32).unsqueeze(1).unsqueeze(0)
    return x_tensor

def generate_synthetic_input(num_steps, threshold=1000.0, pattern="dirac", output_folder=".", noise_mean=0.5, noise_std=0.1, constant_value=0.5):

    """
    Generates a synthetic 1D time series.
    
    For the "steps" pattern the signal alternates between the threshold value and 0.0 in blocks.
    For the "dirac" pattern the signal is zero except for a single dirac-delta spike at the first time step.
    
    Saves a high-quality PNG plot in output_folder.
    
    Returns:
        A tensor of shape (1, num_steps, 1).
    """
    if pattern == "steps":
        block_size = num_steps // 10 if num_steps >= 10 else 10
        data = np.zeros(num_steps, dtype=np.float32)
        for i in range(num_steps):
            block = i // block_size
            data[i] = threshold if (block % 2 == 0) else 0.0

    elif pattern == "dirac":
        data = np.zeros(num_steps, dtype=np.float32)
        data[0] = threshold*2

    elif pattern == "gaussian":
        data = np.random.normal(loc=noise_mean, scale=noise_std, size=num_steps).astype(np.float32)

    elif pattern == "constant":
        data = np.full(num_steps, constant_value, dtype=np.float32)

    elif pattern == "random":
        data = np.random.rand(num_steps).astype(np.float32)

    else:
        raise ValueError("Please select a valid synthetic pattern.")


    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(data, marker='o', linestyle='-')
    ax.set_title("Synthetic Input Time Series")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.grid(True)
    fig.tight_layout()
    synthetic_path = os.path.join(output_folder, "synthetic_input.png")
    fig.savefig(synthetic_path, dpi=600)
    plt.close(fig) 

    x_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(1).unsqueeze(0)
    return x_tensor


class SpikingReservoirLoaded(nn.Module):
    def __init__(self, threshold, beta_reservoir, reservoir_size, device,
                 reset_delay, input_lif_beta, reset_mechanism,
                 connectivity_matrix_path):
        """
        A recurrent spiking reservoir that implements the dynamics manually.
        
        The update equations are:
        
          1. Initialization:
             V(0) = 0, S(0) = 0
             
          2. For each time step t:
             - External Input:
               For t = 0: I_ext = [2 * threshold, 2 * threshold, …, 2 * threshold]
               For t > 0: I_ext = x_t repeated across neurons.
             - Recurrent Input:
               I_rec = S(t) ⋅ W
             - Update Membrane Potential:
               V(t+1) = beta * V(t) + I_ext + I_rec
             - Generate Spikes:
               S(t+1) = Θ(V(t+1) - threshold)
             - Reset:
               V(t+1) = V(t+1) * (1 - S(t+1))
        
        Args:
            threshold (float): Spiking threshold.
            beta_reservoir (float): Decay factor (β).
            reservoir_size (int): Number of neurons.
            device (str): 'cpu' or 'cuda'.
            reset_delay (int): (Not used in this manual implementation.)
            input_lif_beta (float): (Not used in this manual implementation.)
            reset_mechanism (str): (Not used in this manual implementation.)
            connectivity_matrix_path (str): Path to the .npy file containing W.
        """
        set_seed(42)
        super(SpikingReservoirLoaded, self).__init__()
        self.device = device
        self.reservoir_size = reservoir_size
        self.threshold = threshold
        self.beta = beta_reservoir
        
        # Load the connectivity matrix W from file.
        W = torch.tensor(np.load(connectivity_matrix_path), dtype=torch.float32)
        #W = torch.ones(self.reservoir_size, self.reservoir_size)*self.threshold/(2*self.reservoir_size)
        if W.shape != (self.reservoir_size, self.reservoir_size):
            raise ValueError("Loaded connectivity matrix shape does not match reservoir_size.")
        self.W = W.to(self.device)
        
        self.to(self.device)

    def forward(self, x):
        """
        Simulate the reservoir dynamics.
        
        Args:
            x: Input tensor of shape (batch_size, time_steps, 1)
               For t > 0, x provides external input to the reservoir.
        
        Returns:
            avg_firing_rate, spike_record, mem_record.
            - spike_record: (time_steps, batch_size, reservoir_size)
            - mem_record: (time_steps, batch_size, reservoir_size)
        """
        batch_size, time_steps, _ = x.shape
        x = x.to(self.device)
        
        # Initialization: V(0)=0, S(0)=0.
        V_mem = torch.zeros(batch_size, self.reservoir_size, device=self.device)
        Spk = torch.zeros(batch_size, self.reservoir_size, device=self.device)
        
        spike_record = []
        mem_record = []
        
        for t in range(time_steps):
            if t == 0:
                # At time step 0, force external input to be 2*threshold for all neurons.
                I_input = torch.ones(batch_size, 1, device=self.device) * (2 * self.threshold)
                I_input = I_input.repeat(1, self.reservoir_size)
            else:
                # For t > 0, use the provided input (replicated across neurons).
                # x_t = x[:, t, :]  # shape: (batch_size, 1)
                # I_input = x_t.repeat(1, self.reservoir_size)
                #I_input = torch.zeros(batch_size, 1, device=self.device)
                set_seed(42)
                I_input = torch.rand(batch_size, 1, device=self.device)
                I_input = I_input.repeat(1, self.reservoir_size)
            
            # Recurrent input: I_rec = S(t) ⋅ W.
            I_recurrent = torch.matmul(Spk, self.W)
            
            # Update membrane potential with leakage.
            V_mem_new = self.beta * V_mem + I_input + I_recurrent
            
            # Generate spikes: S(t+1) = Θ(V_new - threshold).
            Spk_new = (V_mem_new >= self.threshold).float()
            
            # Reset membrane potential for neurons that fired.
            V_mem_new = V_mem_new * (1 - Spk_new)
            
            # Record the spike and membrane potential.
            spike_record.append(Spk_new.detach().cpu().numpy())
            mem_record.append(V_mem_new.detach().cpu().numpy())
            
            # Update state for the next time step.
            V_mem, Spk = V_mem_new, Spk_new
        
        spike_record = np.array(spike_record)  # (time_steps, batch_size, reservoir_size)
        mem_record = np.array(mem_record)
        avg_firing_rate = spike_record.mean()
        return avg_firing_rate, spike_record, mem_record

def generate_input_signal(V_threshold, time_steps=200):
    """
    Create an input signal for the reservoir simulation.
    
    The input signal is defined as:
      - First time step = 2 * V_threshold (forcing a spike)
      - Remaining time steps = 0.
    
    Returns:
        A tensor of shape (1, time_steps, 1).
    """
    signal = np.zeros(time_steps, dtype=np.float32)
    signal[0] = 2 * V_threshold
    input_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    return input_tensor
