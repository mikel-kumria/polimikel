import torch
from torch.utils.data import Dataset

class TemporalXORDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.seq_len = 30
        self.data = []
        self.labels = []
        v_th = 2.0  # Increased threshold voltage
        for _ in range(num_samples):
            a = torch.randint(0, 2, (1,)).item()
            b = torch.randint(0, 2, (1,)).item()
            xor = a ^ b
            # Initialize sequence: [30 timesteps x 3 input channels]
            seq = torch.zeros((self.seq_len, 3), dtype=torch.float)
            
            # Input neuron 1 (A): first 10 timesteps active, rest 0
            seq[0:10, 0] = v_th if a == 1 else v_th / 2
            
            # Input neuron 2 (B): timesteps 10-20 active, rest 0
            seq[10:20, 1] = v_th if b == 1 else v_th / 2
            
            # Input neuron 3 (R): last 10 timesteps active with v_th
            seq[20:30, 2] = v_th
            
            self.data.append(seq)
            # One-hot label for XOR result
            label = torch.zeros(2, dtype=torch.float)
            label[xor] = 1.0
            self.labels.append(label)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx] 