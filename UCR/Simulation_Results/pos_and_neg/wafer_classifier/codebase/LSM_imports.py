import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import snntorch as snn
import matplotlib.pyplot as plt

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
        data[0] = threshold

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

class SpikingReservoir(nn.Module):
    def __init__(self, threshold, beta_reservoir, reservoir_size, device,
                 reset_delay, input_lif_beta, reset_mechanism,
                 init_weight_a, init_weight_b, spectral_radius):
        """
        Args:
            threshold (float): Threshold for both input and reservoir LIF neurons.
            beta_reservoir (float): β value for reservoir LIF neurons.
            reservoir_size (int): Number of reservoir neurons.
            device (str): 'cpu' or 'cuda'.
            reset_delay (int): Reset delay for neurons.
            input_lif_beta (float): β value for the input LIF neuron.
            reset_mechanism (str): e.g. "zero".
            init_weight_a (float): Lower bound for uniform initialization of recurrent weights.
            init_weight_b (float): Upper bound.
            spectral_radius (float): Desired spectral radius for scaling the recurrent weights.
        """
        super(SpikingReservoir, self).__init__()
        self.device = device
        self.reservoir_size = reservoir_size
        self.input_lif_beta = input_lif_beta

        self.input_fc = nn.Linear(1, 1, bias=False)
        self.input_lif = snn.Leaky(beta=input_lif_beta,
                                   threshold=threshold,
                                   spike_grad=None,
                                   reset_mechanism=reset_mechanism,
                                   reset_delay=reset_delay)
        self.reservoir_fc = nn.Linear(1, reservoir_size, bias=False)
        self.reservoir_lif = snn.RLeaky(beta=beta_reservoir,
                                        linear_features=reservoir_size,
                                        threshold=threshold,
                                        spike_grad=None,
                                        reset_mechanism=reset_mechanism,
                                        reset_delay=reset_delay,
                                        all_to_all=True)
        with torch.no_grad():
            nn.init.uniform_(self.reservoir_lif.recurrent.weight, a=init_weight_a, b=init_weight_b)
            W = self.reservoir_lif.recurrent.weight
            eigenvalues = torch.linalg.eigvals(W)
            current_radius = eigenvalues.abs().max()
            scaling_factor = spectral_radius / current_radius
            W.mul_(scaling_factor)
        self.to(self.device)
    
    def forward(self, x):
        """
        Simulate reservoir dynamics one time step at a time.
        
        Args:
            input_reservoir_type (str): "LIF" or "Vmem" or "pass_through" to select if the input neuron is a spiking LIF (LIF), just the Vmem of the LIF without spikes and reset (Vmem) or a pass-through.
            x (tensor): Input tensor of shape (batch_size, time_steps, 1).
        
        Returns:
            avg_firing_rate (float): Average firing rate (over neurons and time).
            spike_record (np.array): Recorded spikes (shape: time_steps x batch_size x reservoir_size).
            mem_record (np.array): Recorded membrane potentials (same shape).
        """

        #################################################################################################################
        #              "LIF"              #              "Vmem"              #              "pass_through"              #     
        #                                                                                                               #                           
        input_reservoir_type = "Vmem"                                                                                    
        #                                                                                                               #            
        #                                   this level of autism is mine, not ChatGPT                                   #
        #################################################################################################################

        batch_size, time_steps, _ = x.shape
        x = x.to(self.device)
        input_mem = torch.zeros(batch_size, 1, device=self.device)
        reservoir_mem = torch.zeros(batch_size, self.reservoir_size, device=self.device)
        reservoir_spk = torch.zeros(batch_size, self.reservoir_size, device=self.device)
        spike_record = []
        mem_record = []

        for t in range(time_steps):

            if input_reservoir_type == "LIF":
                
                x_t = x[:, t, :]  # shape: (batch_size, 1)
                input_current = self.input_fc(x_t)

                input_spk, input_mem = self.input_lif(input_current, input_mem)
                reservoir_current = self.reservoir_fc(input_spk)

                reservoir_spk, reservoir_mem = self.reservoir_lif(reservoir_current,
                                                                reservoir_spk,
                                                                reservoir_mem)
                spike_record.append(reservoir_spk.detach().cpu().numpy())
                mem_record.append(reservoir_mem.detach().cpu().numpy())

            elif input_reservoir_type == "Vmem":

                x_t = x[:, t, :]  # shape: (batch_size, 1)
                input_current = self.input_fc(x_t)

                input_mem = self.input_lif_beta * input_mem + input_current
                reservoir_current = self.reservoir_fc(input_mem)

                reservoir_spk, reservoir_mem = self.reservoir_lif(reservoir_current,
                                                                reservoir_spk,
                                                                reservoir_mem)
                spike_record.append(reservoir_spk.detach().cpu().numpy())
                mem_record.append(reservoir_mem.detach().cpu().numpy())


            elif input_reservoir_type == "pass_through":

                x_t = x[:, t, :]  # shape: (batch_size, 1)
                input_current = self.input_fc(x_t)

                reservoir_current = self.reservoir_fc(input_current)

                reservoir_spk, reservoir_mem = self.reservoir_lif(reservoir_current,
                                                                reservoir_spk,
                                                                reservoir_mem)
                spike_record.append(reservoir_spk.detach().cpu().numpy())
                mem_record.append(reservoir_mem.detach().cpu().numpy())

            else:
                raise ValueError("Please select a valid input_reservoir_type.")

        spike_record = np.array(spike_record)
        mem_record = np.array(mem_record)
        avg_firing_rate = spike_record.mean()
        return avg_firing_rate, spike_record, mem_record

    def get_recurrent_weights(self):
        """Returns the recurrent weight matrix as a numpy array."""
        return self.reservoir_lif.recurrent.weight.detach().cpu().numpy()