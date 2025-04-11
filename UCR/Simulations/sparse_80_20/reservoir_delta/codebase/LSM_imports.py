import os
import re
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import random

def set_seed(seed=42):
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
    where N is one plus the highest existing trial number.
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

def generate_input_signal(V_threshold, time_steps=50):
    """
    Creates a 50–time–step input signal for the reservoir:
      - At time step 0: value = 2 * V_threshold
      - At time steps 1 to 49: value = 0.
    
    Returns:
        A tensor of shape (1, time_steps, 1).
    """
    signal = np.zeros(time_steps, dtype=np.float32)
    signal[0] = 2 * V_threshold
    input_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    return input_tensor

class SpikingReservoirLoaded(nn.Module):
    def __init__(self, threshold, beta_reservoir, reservoir_size, device,
                 reset_delay, input_lif_beta, reset_mechanism,
                 connectivity_matrix_path, input_weights_path=None):
        """
        A recurrent spiking reservoir with a single input LIF neuron that is projected
        via a nn.Linear onto a reservoir of LIF neurons.
        
        Dynamics per time step:
          V(t+1) = beta_reservoir * V(t) + I_input + I_rec
          S(t+1) = 1 if V(t+1) >= threshold, else 0
          Then, V(t+1) is reset: V(t+1) = V(t+1) * (1 - S(t+1))
        
        Args:
          - threshold (float): Spiking threshold.
          - beta_reservoir (float): Leak factor.
          - reservoir_size (int): Number of reservoir neurons.
          - device (str): 'cpu' or 'cuda'.
          - connectivity_matrix_path (str): Path to .npy file for recurrent connectivity W (shape: (reservoir_size, reservoir_size)).
          - input_weights_path (str or None): Path to .npy file for input layer weights (shape: (reservoir_size, 1)); if None, an error is raised.
        """
        set_seed(42)
        super(SpikingReservoirLoaded, self).__init__()
        self.device = device
        self.reservoir_size = reservoir_size
        self.threshold = threshold
        self.beta = beta_reservoir
        self.reset_mechanism = reset_mechanism

        # Load recurrent connectivity matrix.
        W = torch.tensor(np.load(connectivity_matrix_path), dtype=torch.float32)
        if W.shape != (reservoir_size, reservoir_size):
            raise ValueError("Connectivity matrix shape does not match reservoir_size.")
        self.W = W.to(self.device)
        
        # Create input projection layer: maps 1D input to reservoir_size.
        self.input_layer = nn.Linear(1, reservoir_size, bias=False)
        if input_weights_path is not None and os.path.exists(input_weights_path):
            w_in = np.load(input_weights_path)
            w_in = np.array(w_in)
            if w_in.ndim == 1:
                w_in = w_in.reshape(reservoir_size, 1)
            elif w_in.shape != (reservoir_size, 1):
                raise ValueError("Input weights matrix has incorrect shape.")
            self.input_layer.weight.data = torch.tensor(w_in, dtype=torch.float32, device=self.device)
        else:
            raise ValueError("Input weights path is None or does not exist.")
        
        self.to(self.device)

    def forward(self, x):
        """
        Simulates the reservoir dynamics for input x.
        
        Args:
            x: Tensor of shape (batch_size, time_steps, 1).
        
        Returns:
            avg_firing_rate: scalar average firing rate.
            spike_record: NumPy array of shape (time_steps, batch_size, reservoir_size) (spike outputs).
            mem_record: NumPy array of shape (time_steps, batch_size, reservoir_size) (membrane potentials).
        """
        batch_size, time_steps, _ = x.shape
        x = x.to(self.device)
        V_mem = torch.zeros(batch_size, self.reservoir_size, device=self.device)
        Spk = torch.zeros(batch_size, self.reservoir_size, device=self.device)
        spike_record = []
        mem_record = []
        
        for t in range(time_steps):
            input_signal = x[:, t, :]  # shape: (batch_size, 1)
            I_input = self.input_layer(input_signal)  # shape: (batch_size, reservoir_size)
            I_recur = torch.matmul(Spk, self.W)
            V_new = self.beta * V_mem + I_input + I_recur
            Spk_new = (V_new >= self.threshold).float()
            if self.reset_mechanism == "zero":
                V_new = V_new * (1 - Spk_new)
            elif self.reset_mechanism == "subtract":
                V_new = V_new - Spk_new * self.threshold
            elif self.reset_mechanism == "none":
                pass
            else:
                raise ValueError("Invalid reset mechanism. Choose 'zero', 'subtract', or 'none'.")
            spike_record.append(Spk_new.detach().cpu().numpy())
            mem_record.append(V_new.detach().cpu().numpy())
            V_mem, Spk = V_new, Spk_new
        
        spike_record = np.array(spike_record)
        mem_record = np.array(mem_record)
        avg_firing_rate = spike_record.mean()
        return avg_firing_rate, spike_record, mem_record
