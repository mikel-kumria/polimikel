import numpy as np
import torch
from torch.utils.data import Dataset

def generate_concatenated_wave(wave_type, frequency, sample_rate, duration, amplitude, std_dev, offset, phase=0):
    """
    Generates a concatenated wave sample and label.
    
    For wave_type "sine": generates a sine wave (with Gaussian noise),
    then a silent segment, then a second sine wave.
    For wave_type "square": generates a square wave (with noise),
    then a silent segment, then a second square wave.
    
    Args:
      wave_type (str): Either "sine" or "square".
      frequency (float): Frequency of the wave.
      sample_rate (int): Samples per second.
      duration (float): Duration (in seconds) of each individual wave segment.
      amplitude (float): Amplitude of the wave.
      std_dev (float): Standard deviation of the noise (multiplied by amplitude).
      offset (float): Offset added to the wave.
      phase (float): Phase of the wave (default 0).
      
    Returns:
      final_wave (np.ndarray): The concatenated wave signal.
      label (int): 0 if sine-sine, 1 if square-square.
    """
    # Time array for one segment.
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    noise = np.random.normal(0, std_dev * amplitude, t.shape)
    
    if wave_type.lower() == "sine":
        wave = amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset + noise
        label = 0
    elif wave_type.lower() == "square":
        wave = amplitude * np.sign(np.sin(2 * np.pi * frequency * t + phase)) + offset + noise
        label = 1
    else:
        raise ValueError("wave_type must be 'sine' or 'square'")
    
    # Create a silent segment (here we use one second of zeros).
    silent_segment = np.zeros(int(sample_rate))
    # Concatenate: wave + silent segment + wave.
    final_wave = np.concatenate([wave, silent_segment, wave])
    return final_wave, label

class ConcatenatedWaveDataset(Dataset):
    """
    A custom dataset that generates samples by concatenating two identical waves
    (either sine or square) with a silent segment in between.
    
    Half of the samples will be sine-sine (label 0) and half will be square-square (label 1).
    
    Args:
      num_samples (int): Number of samples to generate.
      sample_rate (int): Samples per second.
      duration (float): Duration (in seconds) of each wave segment.
      frequency (float): Frequency for the waves.
      amplitude (float): Amplitude of the waves.
      std_dev (float): Standard deviation for noise.
      offset (float): Offset added to the wave.
    """
    def __init__(self, num_samples, sample_rate, duration, frequency, amplitude, std_dev, offset):
        self.samples = []
        self.labels = []
        for i in range(num_samples):
            # For half the samples, use sine; for half, use square.
            if np.random.rand() < 0.5:
                wave, label = generate_concatenated_wave("sine", frequency, sample_rate, duration, amplitude, std_dev, offset)
            else:
                wave, label = generate_concatenated_wave("square", frequency, sample_rate, duration, amplitude, std_dev, offset)
            self.samples.append(wave)
            self.labels.append(label)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        wave = torch.tensor(self.samples[idx], dtype=torch.float)
        label = self.labels[idx]
        return wave, label