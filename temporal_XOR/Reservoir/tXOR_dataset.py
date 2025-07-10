import torch
from torch.utils.data import Dataset
import numpy as np

class TemporalXORDataset(Dataset):
    def __init__(self, num_samples, v_th=2.0, seq_len=None, min_gap=0, max_gap=20, noise_sigma=0.0):
        self.num_samples = num_samples
        self.v_th = v_th
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.noise_sigma = noise_sigma
        
        # Calculate required sequence length based on max_gap
        # 10 timesteps for A + max_gap + 10 timesteps for B + 10 timesteps for R
        required_length = 10 + max_gap + 10 + 10
        self.seq_len = max(seq_len if seq_len is not None else 50, required_length)
        
        self.data = []
        self.labels = []
        
        for _ in range(num_samples):
            a = torch.randint(0, 2, (1,)).item()
            b = torch.randint(0, 2, (1,)).item()
            xor = a ^ b
            # Random gap between neuron 1 and neuron 2
            gap = np.random.randint(self.min_gap, self.max_gap + 1)
            # Calculate indices
            idx_a_start = 0
            idx_a_end = 10
            idx_gap_start = idx_a_end
            idx_gap_end = idx_gap_start + gap
            idx_b_start = idx_gap_end
            idx_b_end = idx_b_start + 10
            idx_r_start = idx_b_end
            idx_r_end = idx_r_start + 10
            # Initialize sequence: [seq_len timesteps x 3 input channels]
            seq = torch.zeros((self.seq_len, 3), dtype=torch.float)
            # Neuron 1 (A): first 10 timesteps
            seq[idx_a_start:idx_a_end, 0] = self.v_th if a == 1 else self.v_th / 2
            # Neuron 2 (B): after gap, for 10 timesteps
            seq[idx_b_start:idx_b_end, 1] = self.v_th if b == 1 else self.v_th / 2
            # Neuron 3 (R): after B, for 10 timesteps
            seq[idx_r_start:idx_r_end, 2] = self.v_th
            
            # Add Gaussian noise to neurons A and B (channels 0 and 1)
            if self.noise_sigma > 0:
                noise = torch.randn_like(seq[:, :2]) * self.noise_sigma
                seq[:, :2] += noise # Add noise to neurons A and B, so only the first 2 channels are noisy
            
            # The rest is already zero (padding)
            self.data.append(seq)
            # One-hot label for XOR result
            label = torch.zeros(2, dtype=torch.float)
            label[xor] = 1.0
            self.labels.append(label)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx] 