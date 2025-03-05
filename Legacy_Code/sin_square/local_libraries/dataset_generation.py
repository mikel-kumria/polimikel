# dataset_generation.py: Contains functions for generating synthetic datasets, specific for the Wave Order task.

if 1==1:
    # imports
    import os
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader

def generate_sinusoidal_wave(frequency, sample_rate, duration, amplitude, std_dev, phase, offset):
    """
    Generates a sinusoidal wave with Gaussian noise.

    Args:
    - frequency (float): Frequency of the wave.
    - sample_rate (int): Number of samples per second.
    - duration (float): Duration of the wave in seconds.
    - amplitude (float): Amplitude of the wave.
    - std_dev (float): Standard deviation of the Gaussian noise.
    - phase (float): Phase of the wave.
    - offset (float): Offset to be added to the wave.

    Returns:
    - np.ndarray: Sinusoidal wave with noise.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t + phase) + offset
    gaussian_noise = np.random.normal(0, std_dev * amplitude, wave.shape)
    return wave + gaussian_noise

def generate_square_wave(frequency, sample_rate, duration, amplitude, std_dev, phase, offset):
    """
    Generates a square wave with Gaussian noise.

    Args:
    - frequency (float): Frequency of the wave.
    - sample_rate (int): Number of samples per second.
    - duration (float): Duration of the wave in seconds.
    - amplitude (float): Amplitude of the wave.
    - std_dev (float): Standard deviation of the Gaussian noise.
    - phase (float): Phase of the wave.
    - offset (float): Offset to be added to the wave.

    Returns:
    - np.ndarray: Square wave with noise.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sign(np.sin(2 * np.pi * frequency * t + phase)) + offset
    gaussian_noise = np.random.normal(0, std_dev * amplitude, wave.shape)
    return wave + gaussian_noise

class Wave_Order_Dataset(Dataset):
    """
    Custom Dataset for generating waves with different parameters.

    Args:
    - num_samples (int): Number of samples in the dataset.
    - sample_rate (int): Number of samples per second.
    - duration (float): Duration of each wave in seconds.
    - freq_min (float): Minimum frequency of the waves.
    - freq_max (float): Maximum frequency of the waves.
    - amp_min (float): Minimum amplitude of the waves.
    - amp_max (float): Maximum amplitude of the waves.
    - std_dev (float): Standard deviation of the Gaussian noise.
    - offset (float): Offset to be added to the waves.
    """
    def __init__(self, num_samples, sample_rate, duration, freq_min, freq_max, amp_min, amp_max, std_dev, offset):
        self.samples = []

        for _ in range(num_samples):
            frequency1 = np.random.uniform(freq_min, freq_max)
            amplitude1 = np.random.uniform(amp_min, amp_max)
            wave_type1 = np.random.choice(['sine', 'square'])
            #phase1 = 0  # 
            phase1 = np.random.uniform(0, 2 * np.pi)

            frequency2 = np.random.uniform(freq_min, freq_max)
            amplitude2 = np.random.uniform(amp_min, amp_max)
            wave_type2 = np.random.choice(['sine', 'square'])
            #phase2 = 0  # 
            phase2 = np.random.uniform(0, 2 * np.pi)

            if wave_type1 == 'sine':
                wave1 = generate_sinusoidal_wave(frequency1, sample_rate, 1, amplitude1, std_dev, phase1, offset)
                label1 = 0
            else:
                wave1 = generate_square_wave(frequency1, sample_rate, 1, amplitude1, std_dev, phase1, offset)
                label1 = 1

            if wave_type2 == 'sine':
                wave2 = generate_sinusoidal_wave(frequency2, sample_rate, 1, amplitude2, std_dev, phase2, offset)
                label2 = 0
            else:
                wave2 = generate_square_wave(frequency2, sample_rate, 1, amplitude2, std_dev, phase2, offset)
                label2 = 1

            if (label1 == 0 and label2 == 1) or (label1 == 1 and label2 == 0):
                combined_wave = np.concatenate((wave1, np.zeros(sample_rate) * offset, wave2))  # np.zeros if you don't want offset, np.ones if you want offset
                combined_label = 0 if (label1 == 0 and label2 == 1) else 1

                self.samples.append((combined_wave, combined_label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        wave, label = self.samples[idx]
        return torch.tensor(wave, dtype=torch.float), label

def split_dataset(dataset, train_ratio, validation_ratio, test_ratio):
    """
    Splits the dataset into training, validation, and test sets.

    Args:
    - dataset (Dataset): The dataset to be split.
    - train_ratio (float): Ratio of the training set.
    - validation_ratio (float): Ratio of the validation set.
    - test_ratio (float): Ratio of the test set.

    Returns:
    - tuple: (train_dataset, validation_dataset, test_dataset)
    """
    dataset_size = len(dataset)
    train_size = int(dataset_size * train_ratio)
    validation_size = int(dataset_size * validation_ratio)
    test_size = dataset_size - train_size - validation_size

    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, validation_size, test_size])
    return train_dataset, validation_dataset, test_dataset

def get_dataloaders(train_dataset, validation_dataset, test_dataset, batch_size):
    """
    Creates DataLoader objects for training, validation, and test sets.

    Args:
    - train_dataset (Dataset): The training dataset.
    - validation_dataset (Dataset): The validation dataset.
    - test_dataset (Dataset): The test dataset.
    - batch_size (int): Batch size for the DataLoader.

    Returns:
    - tuple: (train_loader, validation_loader, test_loader)
    """
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, validation_loader, test_loader