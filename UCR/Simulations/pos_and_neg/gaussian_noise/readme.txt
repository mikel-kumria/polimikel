Input signal: gaussian random noise
Mean used: 0.0
std used: 1.0


NOTE: to change the kind of input used for the reservoir (LIF, Pass-Through or Vmem), in the file LSM_imports.py change the variable "input_reservoir_type", so for example:

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
------> input_reservoir_type = "LIF" or "Vmem" or "pass_through"
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
