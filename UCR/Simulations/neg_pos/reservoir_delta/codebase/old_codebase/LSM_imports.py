import os
import torch
import torch.nn as nn
import snntorch as snn
import numpy as np
import random

class SpikingReservoirLoaded(nn.Module):
    def __init__(self, threshold, beta_reservoir, reservoir_size, device,
                 reset_delay, input_lif_beta, reset_mechanism,
                 connectivity_matrix_path):
        """
        A reservoir of spiking LIF neurons that loads its recurrent connectivity
        matrix from a file and directly injects the input signal into the reservoir neurons.
        
        Args:
            threshold (float): The spiking threshold (V_threshold). This value is directly injected at time step 0.
            beta_reservoir (float): The decay factor for reservoir neurons.
            reservoir_size (int): Number of reservoir neurons.
            device (str): 'cpu' or 'cuda'.
            reset_delay (int): Reset delay for neurons.
            input_lif_beta (float): (Not used now; provided for compatibility.)
            reset_mechanism (str): e.g. "zero".
            connectivity_matrix_path (str): Path to the .npy file for the recurrent connectivity matrix.
        """
        set_seed(42)
        super(SpikingReservoirLoaded, self).__init__()

        self.device = device
        self.reservoir_size = reservoir_size

        self.reservoir_lif = snn.RLeaky(beta=beta_reservoir,
                                        linear_features=reservoir_size,
                                        threshold=threshold,
                                        spike_grad=None,
                                        reset_mechanism=reset_mechanism,
                                        reset_delay=reset_delay,
                                        all_to_all=True)
        # Connectivity matrix is loaded from the path which I specify
        W = torch.tensor(np.load(connectivity_matrix_path), dtype=torch.float32)

        #W = torch.full((reservoir_size, reservoir_size), threshold/10, dtype=torch.float32)

        if W.shape != (reservoir_size, reservoir_size):
            raise ValueError("Loaded connectivity matrix shape does not match reservoir_size.")
        with torch.no_grad():
            self.reservoir_lif.recurrent.weight.copy_(W)
        self.to(self.device)

    def forward(self, x):
        """
        Simulate reservoir dynamics.
        x: Input tensor of shape (batch_size, time_steps, 1)
        Returns:
            avg_firing_rate, spike_record, mem_record.
        """
        # Direct injection mode (no input LIF, no scaling).
        batch_size, time_steps, _ = x.shape
        x = x.to(self.device)
        # Initialize the reservoir membrane potential and spike state.
        reservoir_mem = torch.zeros(batch_size, self.reservoir_size, device=self.device)
        reservoir_spk = torch.zeros(batch_size, self.reservoir_size, device=self.device)
        spike_record = []
        mem_record = []

        for t in range(time_steps):
            # Directly replicate the input scalar to all reservoir neurons.
            x_t = x[:, t, :]  # shape: (batch_size, 1)
            input_current = x_t.repeat(1, self.reservoir_size)
            reservoir_spk, reservoir_mem = self.reservoir_lif(input_current, reservoir_spk, reservoir_mem)
            spike_record.append(reservoir_spk.detach().cpu().numpy())
            mem_record.append(reservoir_mem.detach().cpu().numpy())

        spike_record = np.array(spike_record)  # shape: (time_steps, batch_size, reservoir_size)
        mem_record = np.array(mem_record)
        avg_firing_rate = spike_record.mean()
        return avg_firing_rate, spike_record, mem_record

def generate_input_signal(V_threshold, time_steps=200):
    """
    Create an input signal:
      - First time step = V_threshold (forcing a spike)
      - Remaining time steps = 0.
    Returns a tensor of shape (1, time_steps, 1).
    """
    signal = np.zeros(time_steps, dtype=np.float32)
    signal[0] = V_threshold*1.1
    input_tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    return input_tensor

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
