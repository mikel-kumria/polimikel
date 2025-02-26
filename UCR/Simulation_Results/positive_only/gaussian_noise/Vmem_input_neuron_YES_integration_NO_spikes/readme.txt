Here we have used this forward, which takes the input current and accumulates the Vmem on the neuron without ever spiking (no reset), and passes this Vmem to the reservoir:

for t in range(time_steps):

            x_t = x[:, t, :]
            input_current = self.input_fc(x_t)
            
            input_mem = input_mem + input_current  # accumulate Vmem over time 
            reservoir_current = self.reservoir_fc(input_mem)

            
            reservoir_spk, reservoir_mem = self.reservoir_lif(reservoir_current,reservoir_spk,reservoir_mem)
            
            spike_record.append(reservoir_spk.detach().cpu().numpy())
            mem_record.append(reservoir_mem.detach().cpu().numpy())
            
            


Instead of the classic LIF neuron, which generates spikes given on the input:

for t in range(time_steps):

	    x_t = x[:, t, :]  # shape: (batch_size, 1)
            input_current = self.input_fc(x_t)
            input_spk, input_mem = self.input_lif(input_current, input_mem)
            reservoir_current = self.reservoir_fc(input_spk)

            reservoir_spk, reservoir_mem = self.reservoir_lif(reservoir_current,reservoir_spk,reservoir_mem)
            
            spike_record.append(reservoir_spk.detach().cpu().numpy())
            mem_record.append(reservoir_mem.detach().cpu().numpy())
