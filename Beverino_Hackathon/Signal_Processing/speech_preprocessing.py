"""
Speech Preprocessing Module for Google Speech Commands Dataset

This module provides 4 different preprocessing methods:
1. MEL Spectrogram
2. MFCC (Mel-frequency cepstral coefficients)
3. Gammatone Filterbank
4. Cochleagram (Auditory-inspired representation)

Each method converts audio to spike-compatible input for SNN.
"""

import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
import librosa
from scipy import signal
from typing import List, Tuple, Optional


class SpeechPreprocessor:
    """
    Base class for speech preprocessing methods
    """
    
    def __init__(self, sample_rate=16000, n_mels=16, n_mfcc=16, n_fft=1024, 
                 hop_length=512, win_length=1024, f_min=0, f_max=8000):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        
    def preprocess(self, audio: torch.Tensor) -> torch.Tensor:
        """Base preprocessing method - to be implemented by subclasses"""
        raise NotImplementedError


class MELPreprocessor(SpeechPreprocessor):
    """
    MEL Spectrogram preprocessing
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mel_transform = T.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            n_mels=self.n_mels,
            f_min=self.f_min,
            f_max=self.f_max,
            mel_scale='htk'
        )
        
    def preprocess(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert audio to MEL spectrogram
        
        Args:
            audio: Audio tensor of shape [channels, samples]
            
        Returns:
            mel_spec: MEL spectrogram of shape [n_mels, time_steps]
        """
        # Ensure audio is mono
        if audio.dim() > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Apply MEL spectrogram
        mel_spec = self.mel_transform(audio)
        
        # Convert to log scale
        mel_spec = torch.log(mel_spec + 1e-9)
        
        # Normalize
        mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-9)
        
        return mel_spec.squeeze(0)  # Remove channel dimension


class MFCCPreprocessor(SpeechPreprocessor):
    """
    MFCC (Mel-frequency cepstral coefficients) preprocessing
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mfcc_transform = T.MFCC(
            sample_rate=self.sample_rate,
            n_mfcc=self.n_mfcc,
            melkwargs={
                'n_fft': self.n_fft,
                'hop_length': self.hop_length,
                'win_length': self.win_length,
                'n_mels': self.n_mels,
                'f_min': self.f_min,
                'f_max': self.f_max,
                'mel_scale': 'htk'
            }
        )
        
    def preprocess(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert audio to MFCC
        
        Args:
            audio: Audio tensor of shape [channels, samples]
            
        Returns:
            mfcc: MFCC features of shape [n_mfcc, time_steps]
        """
        # Ensure audio is mono
        if audio.dim() > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        # Apply MFCC
        mfcc = self.mfcc_transform(audio)
        
        # Normalize
        mfcc = (mfcc - mfcc.mean()) / (mfcc.std() + 1e-9)
        
        return mfcc.squeeze(0)  # Remove channel dimension


class GammatonePreprocessor(SpeechPreprocessor):
    """
    Gammatone Filterbank preprocessing (auditory-inspired)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create gammatone filterbank center frequencies (logarithmically spaced)
        self.center_freqs = torch.logspace(
            torch.log10(torch.tensor(self.f_min)), 
            torch.log10(torch.tensor(self.f_max)), 
            self.n_mels
        )
        
    def gammatone_filter(self, audio: torch.Tensor, center_freq: float) -> torch.Tensor:
        """
        Apply a single gammatone filter
        
        Args:
            audio: Audio tensor
            center_freq: Center frequency of the filter
            
        Returns:
            filtered: Filtered audio
        """
        # Convert to numpy for scipy
        audio_np = audio.numpy()
        
        # Gammatone filter parameters
        order = 4
        bandwidth = 1.019 * center_freq  # Equivalent rectangular bandwidth
        
        # Design gammatone filter
        b, a = signal.gammatone(center_freq, order, self.sample_rate, bandwidth)
        
        # Apply filter
        filtered = signal.filtfilt(b, a, audio_np)
        
        return torch.tensor(filtered, dtype=torch.float32)
        
    def preprocess(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert audio to gammatone filterbank representation
        
        Args:
            audio: Audio tensor of shape [channels, samples]
            
        Returns:
            gammatone_spec: Gammatone spectrogram of shape [n_mels, time_steps]
        """
        # Ensure audio is mono
        if audio.dim() > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        audio = audio.squeeze(0)  # Remove channel dimension
        
        # Apply gammatone filters
        filtered_signals = []
        for center_freq in self.center_freqs:
            filtered = self.gammatone_filter(audio, center_freq.item())
            filtered_signals.append(filtered)
        
        # Stack filtered signals
        gammatone_spec = torch.stack(filtered_signals, dim=0)
        
        # Apply envelope extraction (Hilbert transform)
        gammatone_spec = torch.abs(torch.fft.hilbert(gammatone_spec))
        
        # Downsample to reduce temporal resolution
        downsample_factor = len(audio) // (self.n_fft // self.hop_length)
        if downsample_factor > 1:
            gammatone_spec = gammatone_spec[:, ::downsample_factor]
        
        # Convert to log scale
        gammatone_spec = torch.log(gammatone_spec + 1e-9)
        
        # Normalize
        gammatone_spec = (gammatone_spec - gammatone_spec.mean()) / (gammatone_spec.std() + 1e-9)
        
        return gammatone_spec


class CochleagramPreprocessor(SpeechPreprocessor):
    """
    Cochleagram preprocessing (cochlea-inspired representation)
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create cochlear filterbank center frequencies (logarithmically spaced)
        self.center_freqs = torch.logspace(
            torch.log10(torch.tensor(self.f_min)), 
            torch.log10(torch.tensor(self.f_max)), 
            self.n_mels
        )
        
    def cochlear_filter(self, audio: torch.Tensor, center_freq: float) -> torch.Tensor:
        """
        Apply a single cochlear filter (simplified)
        
        Args:
            audio: Audio tensor
            center_freq: Center frequency of the filter
            
        Returns:
            filtered: Filtered audio
        """
        # Convert to numpy for scipy
        audio_np = audio.numpy()
        
        # Cochlear filter parameters (simplified)
        q = 8  # Quality factor
        bandwidth = center_freq / q
        
        # Design bandpass filter
        b, a = signal.butter(4, [center_freq - bandwidth/2, center_freq + bandwidth/2], 
                           btype='band', fs=self.sample_rate)
        
        # Apply filter
        filtered = signal.filtfilt(b, a, audio_np)
        
        return torch.tensor(filtered, dtype=torch.float32)
        
    def preprocess(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Convert audio to cochleagram representation
        
        Args:
            audio: Audio tensor of shape [channels, samples]
            
        Returns:
            cochleagram: Cochleagram of shape [n_mels, time_steps]
        """
        # Ensure audio is mono
        if audio.dim() > 1:
            audio = audio.mean(dim=0, keepdim=True)
        
        audio = audio.squeeze(0)  # Remove channel dimension
        
        # Apply cochlear filters
        filtered_signals = []
        for center_freq in self.center_freqs:
            filtered = self.cochlear_filter(audio, center_freq.item())
            filtered_signals.append(filtered)
        
        # Stack filtered signals
        cochleagram = torch.stack(filtered_signals, dim=0)
        
        # Apply half-wave rectification (like hair cells)
        cochleagram = torch.relu(cochleagram)
        
        # Apply compression (like inner hair cells)
        cochleagram = torch.log(cochleagram + 1e-9)
        
        # Downsample to reduce temporal resolution
        downsample_factor = len(audio) // (self.n_fft // self.hop_length)
        if downsample_factor > 1:
            cochleagram = cochleagram[:, ::downsample_factor]
        
        # Normalize
        cochleagram = (cochleagram - cochleagram.mean()) / (cochleagram.std() + 1e-9)
        
        return cochleagram


class SpeechCommandsDataset(Dataset):
    """
    Dataset for Google Speech Commands with preprocessing
    """
    
    def __init__(self, root_dir: str, commands: List[str], preprocessing: str = "mel",
                 spike_encoding: bool = True, max_length: int = 16000, 
                 transform=None, subset: str = "training"):
        """
        Args:
            root_dir: Path to Google Speech Commands dataset
            commands: List of command words to include
            preprocessing: Preprocessing method ("mel", "mfcc", "gammatone", "cochleagram")
            spike_encoding: Whether to encode as spikes
            max_length: Maximum audio length in samples
            transform: Additional transforms
            subset: Dataset subset ("training", "validation", "testing")
        """
        self.root_dir = root_dir
        self.commands = commands
        self.preprocessing = preprocessing
        self.spike_encoding = spike_encoding
        self.max_length = max_length
        self.transform = transform
        self.subset = subset
        
        # Create label mapping
        self.label_to_idx = {cmd: idx for idx, cmd in enumerate(commands)}
        
        # Initialize preprocessor
        if preprocessing == "mel":
            self.preprocessor = MELPreprocessor()
        elif preprocessing == "mfcc":
            self.preprocessor = MFCCPreprocessor()
        elif preprocessing == "gammatone":
            self.preprocessor = GammatonePreprocessor()
        elif preprocessing == "cochleagram":
            self.preprocessor = CochleagramPreprocessor()
        else:
            raise ValueError(f"Unknown preprocessing method: {preprocessing}")
        
        # Load file paths
        self.file_paths = []
        self.labels = []
        
        for cmd in commands:
            cmd_dir = os.path.join(root_dir, cmd)
            if os.path.exists(cmd_dir):
                for filename in os.listdir(cmd_dir):
                    if filename.endswith('.wav'):
                        self.file_paths.append(os.path.join(cmd_dir, filename))
                        self.labels.append(self.label_to_idx[cmd])
        
        print(f"Loaded {len(self.file_paths)} audio files for {len(commands)} commands")
        
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        # Load audio
        waveform, sample_rate = torchaudio.load(self.file_paths[idx])
        
        # Resample if necessary
        if sample_rate != 16000:
            resampler = T.Resample(sample_rate, 16000)
            waveform = resampler(waveform)
        
        # Pad or truncate to max_length
        if waveform.size(1) > self.max_length:
            waveform = waveform[:, :self.max_length]
        else:
            # Pad with zeros
            padding = self.max_length - waveform.size(1)
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Apply preprocessing
        features = self.preprocessor.preprocess(waveform)
        
        # Convert to spike encoding if requested
        if self.spike_encoding:
            features = self._encode_spikes(features)
        
        # Create one-hot encoded label
        label = torch.zeros(len(self.commands))
        label[self.labels[idx]] = 1.0
        
        # Apply additional transforms
        if self.transform:
            features = self.transform(features)
        
        return features, label
    
    def _encode_spikes(self, features: torch.Tensor) -> torch.Tensor:
        """
        Convert features to spike encoding using rate coding
        
        Args:
            features: Input features of shape [n_features, time_steps]
            
        Returns:
            spikes: Spike encoded features of shape [time_steps, n_features]
        """
        # Normalize features to [0, 1] range
        features = (features - features.min()) / (features.max() - features.min() + 1e-9)
        
        # Convert to spike rates (0 to 1)
        spike_rates = features
        
        # Transpose to [time_steps, n_features] for SNN input
        spike_rates = spike_rates.t()
        
        return spike_rates


def create_speech_dataloaders(root_dir: str, commands: List[str], preprocessing: str = "mel",
                             spike_encoding: bool = True, batch_size: int = 32,
                             train_split: float = 0.8, val_split: float = 0.1,
                             num_workers: int = 4) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test dataloaders for speech commands
    
    Args:
        root_dir: Path to Google Speech Commands dataset
        commands: List of command words to include
        preprocessing: Preprocessing method
        spike_encoding: Whether to encode as spikes
        batch_size: Batch size for dataloaders
        train_split: Fraction of data for training
        val_split: Fraction of data for validation
        num_workers: Number of workers for dataloading
        
    Returns:
        train_loader, val_loader, test_loader: DataLoaders for each split
    """
    
    # Create full dataset
    full_dataset = SpeechCommandsDataset(
        root_dir=root_dir,
        commands=commands,
        preprocessing=preprocessing,
        spike_encoding=spike_encoding
    )
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_split * total_size)
    val_size = int(val_split * total_size)
    test_size = total_size - train_size - val_size
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Created dataloaders:")
    print(f"  Training: {len(train_loader)} batches ({len(train_dataset)} samples)")
    print(f"  Validation: {len(val_loader)} batches ({len(val_dataset)} samples)")
    print(f"  Testing: {len(test_loader)} batches ({len(test_dataset)} samples)")
    
    return train_loader, val_loader, test_loader


def get_preprocessing_info(preprocessing: str) -> dict:
    """
    Get information about preprocessing methods
    
    Args:
        preprocessing: Preprocessing method name
        
    Returns:
        info: Dictionary with preprocessing information
    """
    info = {
        "mel": {
            "description": "MEL Spectrogram - Standard audio feature extraction",
            "advantages": ["Widely used", "Good frequency resolution", "Fast computation"],
            "disadvantages": ["Linear frequency scale", "Not biologically inspired"]
        },
        "mfcc": {
            "description": "Mel-frequency cepstral coefficients - Compact representation",
            "advantages": ["Compact features", "Good for classification", "Standard in speech recognition"],
            "disadvantages": ["Loses phase information", "Not biologically inspired"]
        },
        "gammatone": {
            "description": "Gammatone Filterbank - Auditory-inspired frequency analysis",
            "advantages": ["Biologically inspired", "Good frequency resolution", "Matches human hearing"],
            "disadvantages": ["Computationally expensive", "Complex implementation"]
        },
        "cochleagram": {
            "description": "Cochleagram - Cochlea-inspired representation",
            "advantages": ["Most biologically realistic", "Includes hair cell dynamics", "Good temporal resolution"],
            "disadvantages": ["Very computationally expensive", "Complex parameters"]
        }
    }
    
    return info.get(preprocessing, {"description": "Unknown preprocessing method"}) 