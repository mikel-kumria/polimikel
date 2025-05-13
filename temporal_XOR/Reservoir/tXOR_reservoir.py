import os
import torch
import torch.nn as nn
import numpy as np
from LSM_imports import set_seed, scale_matrix_to_radius

class TemporalXORReservoirNetwork(nn.Module):
    """
    Temporal XOR using a spiking reservoir in the hidden layer.
    Input: [seq_len, batch_size, input_size] (here input_size=3)
    Output: spk_rec [seq_len, batch_size, reservoir_size], logits [batch_size, output_size]

    Hyperparameters:
      - input_size: number of input channels (3)
      - reservoir_size: number of reservoir neurons
      - output_size: number of classes (2)
      - threshold: reservoir spike threshold
      - beta_reservoir: reservoir leak factor
      - spectral_radius: for scaling the recurrent weight matrix
      - connectivity_matrix_path: .npy file of base W (reservoir_size x reservoir_size)
      - reset_mechanism: 'zero'|'subtract'|'none'
    """
    def __init__(
        self,
        input_size: int,
        reservoir_size: int,
        output_size: int,
        threshold: float,
        beta_reservoir: float,
        spectral_radius: float,
        connectivity_matrix_path: str,
        reset_mechanism: str = 'zero',
        device: torch.device = torch.device('cpu')
    ):
        super().__init__()
        set_seed(42)
        self.device = device
        self.res_size = reservoir_size
        self.threshold = threshold
        self.beta = beta_reservoir
        self.reset_mechanism = reset_mechanism

        # Load and rescale reservoir connectivity
        W = np.load(connectivity_matrix_path)
        if W.shape != (reservoir_size, reservoir_size):
            raise ValueError(f"W shape {W.shape} != ({reservoir_size},{reservoir_size})")
        W = scale_matrix_to_radius(W, spectral_radius)
        self.W = torch.tensor(W, dtype=torch.float32, device=self.device)

        # Input projection: map input_size -> reservoir_size
        self.input_layer = nn.Linear(input_size, reservoir_size, bias=False)
        nn.init.xavier_uniform_(self.input_layer.weight, gain=1.0)

        # Readout layer: map reservoir -> output_size
        self.readout = nn.Linear(reservoir_size, output_size, bias=False)
        nn.init.xavier_uniform_(self.readout.weight, gain=1.0)

        self.to(self.device)

    def forward(self, x: torch.Tensor):
        # x: [seq_len, batch_size, input_size]
        seq_len, batch, in_size = x.shape
        x = x.permute(1, 0, 2).to(self.device)  # [batch, seq_len, input_size]

        # Initialize reservoir state
        V = torch.zeros(batch, self.res_size, device=self.device)
        S = torch.zeros(batch, self.res_size, device=self.device)

        spk_rec = []  # to store spikes
        # Simulate reservoir
        for t in range(seq_len):
            inp = x[:, t, :]                      # [batch, input_size]
            I_in = self.input_layer(inp)         # [batch, res_size]
            I_rec = S @ self.W                   # [batch, res_size]

            V_new = self.beta * V + I_in + I_rec
            S_new = (V_new >= self.threshold).float()

            # Reset mechanism
            if self.reset_mechanism == 'zero':
                V_new = V_new * (1 - S_new)
            elif self.reset_mechanism == 'subtract':
                V_new = V_new - S_new * self.threshold
            # else: no reset

            spk_rec.append(S_new)
            V, S = V_new, S_new

        spk_rec = torch.stack(spk_rec, dim=0)  # [seq_len, batch, res_size]

        # Average over last 10 timesteps
        spk_avg = spk_rec[-10:].mean(dim=0)   # [batch, res_size]

        # Readout logits
        logits = self.readout(spk_avg)        # [batch, output_size]
        return spk_rec, logits