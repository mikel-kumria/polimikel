"""
Comprehensive Visualization Module for SNN Speech Recognition System

This module provides detailed visualizations for:
1. Audio preprocessing (original audio, spectrograms, processed features)
2. LIF neuron firing patterns (raster plots, spike timing)
3. Layer-by-layer activity (membrane potentials, spike rates, heatmaps)
4. Network dynamics (statistics, performance metrics)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union
import librosa
import librosa.display
from scipy import signal
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class SNNVisualizer:
    """
    Comprehensive visualizer for SNN speech recognition system
    """
    
    def __init__(self, figsize=(15, 10), dpi=100):
        self.figsize = figsize
        self.dpi = dpi
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
    def plot_audio_preprocessing(self, audio: torch.Tensor, preprocessed_features: torch.Tensor,
                                sample_rate: int = 16000, preprocessing_type: str = "mel",
                                title: str = "Audio Preprocessing Pipeline"):
        """
        Plot the complete audio preprocessing pipeline
        
        Args:
            audio: Original audio tensor [channels, samples]
            preprocessed_features: Processed features [n_features, time_steps]
            sample_rate: Audio sample rate
            preprocessing_type: Type of preprocessing used
            title: Plot title
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(4, 2, figure=fig, height_ratios=[1, 1, 1, 1], width_ratios=[1, 1])
        
        # 1. Original audio waveform
        ax1 = fig.add_subplot(gs[0, :])
        time = np.arange(len(audio.squeeze())) / sample_rate
        ax1.plot(time, audio.squeeze(), color='blue', alpha=0.7, linewidth=0.8)
        ax1.set_title("Original Audio Waveform", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)
        
        # 2. Audio spectrogram
        ax2 = fig.add_subplot(gs[1, 0])
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio.squeeze().numpy())), ref=np.max)
        librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sample_rate, ax=ax2)
        ax2.set_title("Audio Spectrogram", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Frequency (Hz)")
        
        # 3. Preprocessed features heatmap
        ax3 = fig.add_subplot(gs[1, 1])
        im = ax3.imshow(preprocessed_features.numpy(), aspect='auto', cmap='viridis', 
                       origin='lower', interpolation='nearest')
        ax3.set_title(f"{preprocessing_type.upper()} Features", fontsize=12, fontweight='bold')
        ax3.set_xlabel("Time Steps")
        ax3.set_ylabel("Feature Channels")
        plt.colorbar(im, ax=ax3, label="Feature Value")
        
        # 4. Feature statistics
        ax4 = fig.add_subplot(gs[2, 0])
        feature_means = preprocessed_features.mean(dim=1)
        feature_stds = preprocessed_features.std(dim=1)
        channels = range(len(feature_means))
        
        ax4.bar(channels, feature_means.numpy(), yerr=feature_stds.numpy(), 
               alpha=0.7, color='skyblue', capsize=3)
        ax4.set_title("Feature Statistics", fontsize=12, fontweight='bold')
        ax4.set_xlabel("Feature Channel")
        ax4.set_ylabel("Mean ± Std")
        ax4.grid(True, alpha=0.3)
        
        # 5. Time evolution of features
        ax5 = fig.add_subplot(gs[2, 1])
        time_steps = preprocessed_features.shape[1]
        for i in range(min(8, preprocessed_features.shape[0])):  # Plot first 8 features
            ax5.plot(range(time_steps), preprocessed_features[i].numpy(), 
                    label=f'Channel {i}', alpha=0.8, linewidth=1)
        ax5.set_title("Feature Time Evolution", fontsize=12, fontweight='bold')
        ax5.set_xlabel("Time Steps")
        ax5.set_ylabel("Feature Value")
        ax5.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax5.grid(True, alpha=0.3)
        
        # 6. Feature distribution
        ax6 = fig.add_subplot(gs[3, :])
        feature_flat = preprocessed_features.flatten().numpy()
        ax6.hist(feature_flat, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
        ax6.axvline(feature_flat.mean(), color='red', linestyle='--', 
                   label=f'Mean: {feature_flat.mean():.3f}')
        ax6.axvline(feature_flat.std(), color='orange', linestyle='--', 
                   label=f'Std: {feature_flat.std():.3f}')
        ax6.set_title("Feature Distribution", fontsize=12, fontweight='bold')
        ax6.set_xlabel("Feature Value")
        ax6.set_ylabel("Frequency")
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    def plot_lif_neuron_firing(self, spike_recordings: Dict[str, torch.Tensor], 
                              membrane_recordings: Dict[str, torch.Tensor],
                              layer_names: List[str] = None,
                              max_neurons_per_layer: int = 20,
                              time_range: Tuple[int, int] = None):
        """
        Plot LIF neuron firing patterns with raster plots and membrane potentials
        
        Args:
            spike_recordings: Dictionary with spike recordings for each layer
            membrane_recordings: Dictionary with membrane potential recordings
            layer_names: Names of layers to plot
            max_neurons_per_layer: Maximum number of neurons to show per layer
            time_range: Time range to plot (start, end)
        """
        if layer_names is None:
            layer_names = list(spike_recordings.keys())
        
        n_layers = len(layer_names)
        fig, axes = plt.subplots(n_layers, 2, figsize=(16, 4*n_layers))
        if n_layers == 1:
            axes = axes.reshape(1, -1)
        
        for i, layer_name in enumerate(layer_names):
            spikes = spike_recordings[f'{layer_name}_spikes']
            membrane = membrane_recordings[f'{layer_name}_membrane']
            
            # Limit time range if specified
            if time_range is not None:
                start, end = time_range
                spikes = spikes[start:end]
                membrane = membrane[start:end]
            
            # Limit number of neurons if too many
            n_neurons = min(spikes.shape[1], max_neurons_per_layer)
            spikes = spikes[:, :n_neurons]
            membrane = membrane[:, :n_neurons]
            
            # Raster plot
            ax_raster = axes[i, 0]
            spike_times, neuron_ids = torch.where(spikes > 0)
            if len(spike_times) > 0:
                ax_raster.scatter(spike_times, neuron_ids, s=10, c='red', alpha=0.7)
            ax_raster.set_title(f"{layer_name.title()} Layer - Spike Raster Plot", 
                              fontsize=12, fontweight='bold')
            ax_raster.set_xlabel("Time Steps")
            ax_raster.set_ylabel("Neuron ID")
            ax_raster.set_ylim(-0.5, n_neurons - 0.5)
            ax_raster.grid(True, alpha=0.3)
            
            # Membrane potential heatmap
            ax_mem = axes[i, 1]
            im = ax_mem.imshow(membrane.T.numpy(), aspect='auto', cmap='RdBu_r', 
                             origin='lower', interpolation='nearest')
            ax_mem.set_title(f"{layer_name.title()} Layer - Membrane Potentials", 
                           fontsize=12, fontweight='bold')
            ax_mem.set_xlabel("Time Steps")
            ax_mem.set_ylabel("Neuron ID")
            plt.colorbar(im, ax=ax_mem, label="Membrane Potential")
        
        plt.suptitle("LIF Neuron Firing Patterns", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    def plot_layer_activity_summary(self, spike_recordings: Dict[str, torch.Tensor],
                                   membrane_recordings: Dict[str, torch.Tensor],
                                   layer_names: List[str] = None):
        """
        Plot comprehensive layer activity summary
        
        Args:
            spike_recordings: Dictionary with spike recordings for each layer
            membrane_recordings: Dictionary with membrane potential recordings
            layer_names: Names of layers to plot
        """
        if layer_names is None:
            layer_names = [name.replace('_spikes', '') for name in spike_recordings.keys()]
        
        n_layers = len(layer_names)
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, n_layers, figure=fig, height_ratios=[1, 1, 1])
        
        for i, layer_name in enumerate(layer_names):
            spikes = spike_recordings[f'{layer_name}_spikes']
            membrane = membrane_recordings[f'{layer_name}_membrane']
            
            # 1. Spike rate over time
            ax1 = fig.add_subplot(gs[0, i])
            spike_rate = spikes.mean(dim=1)  # Average across neurons
            ax1.plot(spike_rate.numpy(), color='red', linewidth=2, alpha=0.8)
            ax1.set_title(f"{layer_name.title()} - Spike Rate", fontsize=11, fontweight='bold')
            ax1.set_xlabel("Time Steps")
            ax1.set_ylabel("Spike Rate")
            ax1.grid(True, alpha=0.3)
            
            # 2. Neuron activity distribution
            ax2 = fig.add_subplot(gs[1, i])
            total_spikes_per_neuron = spikes.sum(dim=0)
            ax2.hist(total_spikes_per_neuron.numpy(), bins=20, alpha=0.7, 
                    color='skyblue', edgecolor='black')
            ax2.set_title(f"{layer_name.title()} - Spike Distribution", fontsize=11, fontweight='bold')
            ax2.set_xlabel("Total Spikes per Neuron")
            ax2.set_ylabel("Frequency")
            ax2.grid(True, alpha=0.3)
            
            # 3. Membrane potential statistics
            ax3 = fig.add_subplot(gs[2, i])
            mem_mean = membrane.mean(dim=1)
            mem_std = membrane.std(dim=1)
            time_steps = range(len(mem_mean))
            ax3.fill_between(time_steps, 
                           (mem_mean - mem_std).numpy(), 
                           (mem_mean + mem_std).numpy(), 
                           alpha=0.3, color='green')
            ax3.plot(mem_mean.numpy(), color='green', linewidth=2, alpha=0.8)
            ax3.set_title(f"{layer_name.title()} - Membrane Potential", fontsize=11, fontweight='bold')
            ax3.set_xlabel("Time Steps")
            ax3.set_ylabel("Membrane Potential")
            ax3.grid(True, alpha=0.3)
        
        plt.suptitle("Layer Activity Summary", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    def plot_network_dynamics(self, spike_recordings: Dict[str, torch.Tensor],
                            membrane_recordings: Dict[str, torch.Tensor],
                            training_history: Dict[str, List[float]] = None):
        """
        Plot overall network dynamics and performance
        
        Args:
            spike_recordings: Dictionary with spike recordings for each layer
            membrane_recordings: Dictionary with membrane potential recordings
            training_history: Training history with loss and accuracy
        """
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig, height_ratios=[1, 1, 1])
        
        # 1. Network-wide spike statistics
        ax1 = fig.add_subplot(gs[0, 0])
        layer_names = []
        total_spikes = []
        avg_spike_rates = []
        
        for layer_name, spikes in spike_recordings.items():
            clean_name = layer_name.replace('_spikes', '')
            layer_names.append(clean_name)
            total_spikes.append(spikes.sum().item())
            avg_spike_rates.append(spikes.mean().item())
        
        x = np.arange(len(layer_names))
        width = 0.35
        
        ax1.bar(x - width/2, total_spikes, width, label='Total Spikes', alpha=0.7, color='red')
        ax1_twin = ax1.twinx()
        ax1_twin.bar(x + width/2, avg_spike_rates, width, label='Avg Spike Rate', alpha=0.7, color='blue')
        
        ax1.set_title("Network Spike Statistics", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Layer")
        ax1.set_ylabel("Total Spikes", color='red')
        ax1_twin.set_ylabel("Average Spike Rate", color='blue')
        ax1.set_xticks(x)
        ax1.set_xticklabels(layer_names, rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # 2. Membrane potential evolution
        ax2 = fig.add_subplot(gs[0, 1])
        for layer_name, membrane in membrane_recordings.items():
            clean_name = layer_name.replace('_membrane', '')
            mem_mean = membrane.mean(dim=1)
            ax2.plot(mem_mean.numpy(), label=clean_name, alpha=0.8, linewidth=2)
        
        ax2.set_title("Membrane Potential Evolution", fontsize=12, fontweight='bold')
        ax2.set_xlabel("Time Steps")
        ax2.set_ylabel("Average Membrane Potential")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Spike timing correlation
        ax3 = fig.add_subplot(gs[0, 2])
        # Calculate correlation between layers
        layer_spike_rates = []
        for spikes in spike_recordings.values():
            layer_spike_rates.append(spikes.mean(dim=1).numpy())
        
        if len(layer_spike_rates) > 1:
            corr_matrix = np.corrcoef(layer_spike_rates)
            im = ax3.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax3.set_title("Layer Spike Rate Correlation", fontsize=12, fontweight='bold')
            ax3.set_xticks(range(len(layer_names)))
            ax3.set_yticks(range(len(layer_names)))
            ax3.set_xticklabels(layer_names, rotation=45)
            ax3.set_yticklabels(layer_names)
            plt.colorbar(im, ax=ax3, label="Correlation")
        
        # 4. Training history (if available)
        if training_history is not None:
            ax4 = fig.add_subplot(gs[1, :])
            epochs = range(1, len(training_history.get('train_loss', [])) + 1)
            
            if 'train_loss' in training_history:
                ax4.plot(epochs, training_history['train_loss'], label='Training Loss', 
                        color='blue', linewidth=2)
            if 'val_loss' in training_history:
                ax4.plot(epochs, training_history['val_loss'], label='Validation Loss', 
                        color='red', linewidth=2)
            
            ax4_twin = ax4.twinx()
            if 'train_acc' in training_history:
                ax4_twin.plot(epochs, training_history['train_acc'], label='Training Accuracy', 
                             color='green', linewidth=2, linestyle='--')
            if 'val_acc' in training_history:
                ax4_twin.plot(epochs, training_history['val_acc'], label='Validation Accuracy', 
                             color='orange', linewidth=2, linestyle='--')
            
            ax4.set_title("Training History", fontsize=12, fontweight='bold')
            ax4.set_xlabel("Epoch")
            ax4.set_ylabel("Loss", color='blue')
            ax4_twin.set_ylabel("Accuracy", color='green')
            ax4.legend(loc='upper left')
            ax4_twin.legend(loc='upper right')
            ax4.grid(True, alpha=0.3)
        
        # 5. Neuron activity heatmap
        ax5 = fig.add_subplot(gs[2, :])
        # Combine all layers for overall activity view
        all_activities = []
        for layer_name, spikes in spike_recordings.items():
            clean_name = layer_name.replace('_spikes', '')
            activity = spikes.mean(dim=0)  # Average over time
            all_activities.append(activity.numpy())
        
        combined_activity = np.concatenate(all_activities)
        activity_heatmap = combined_activity.reshape(1, -1)
        
        im = ax5.imshow(activity_heatmap, aspect='auto', cmap='viridis', 
                       interpolation='nearest')
        ax5.set_title("Neuron Activity Heatmap (All Layers)", fontsize=12, fontweight='bold')
        ax5.set_xlabel("Neuron ID")
        ax5.set_ylabel("Layer")
        ax5.set_yticks([])
        
        # Add layer separators
        current_pos = 0
        for i, activity in enumerate(all_activities):
            current_pos += len(activity)
            if i < len(all_activities) - 1:
                ax5.axvline(current_pos - 0.5, color='white', linewidth=2)
        
        plt.colorbar(im, ax=ax5, label="Average Spike Rate")
        
        plt.suptitle("Network Dynamics Overview", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
    def plot_comprehensive_analysis(self, audio: torch.Tensor, preprocessed_features: torch.Tensor,
                                  spike_recordings: Dict[str, torch.Tensor],
                                  membrane_recordings: Dict[str, torch.Tensor],
                                  sample_rate: int = 16000, preprocessing_type: str = "mel",
                                  training_history: Dict[str, List[float]] = None,
                                  title: str = "Complete SNN Speech Processing Analysis"):
        """
        Create a comprehensive analysis plot showing all aspects of the system
        
        Args:
            audio: Original audio tensor
            preprocessed_features: Processed features
            spike_recordings: Spike recordings for all layers
            membrane_recordings: Membrane potential recordings for all layers
            sample_rate: Audio sample rate
            preprocessing_type: Type of preprocessing used
            training_history: Training history (optional)
            title: Overall title
        """
        # Create a large figure with multiple subplots
        fig = plt.figure(figsize=(20, 24))
        gs = GridSpec(6, 4, figure=fig, height_ratios=[1, 1, 1, 1, 1, 1], width_ratios=[1, 1, 1, 1])
        
        # 1. Audio preprocessing (top row)
        # Original audio
        ax1 = fig.add_subplot(gs[0, 0])
        time = np.arange(len(audio.squeeze())) / sample_rate
        ax1.plot(time, audio.squeeze(), color='blue', alpha=0.7, linewidth=0.8)
        ax1.set_title("Original Audio", fontsize=10, fontweight='bold')
        ax1.set_xlabel("Time (s)")
        ax1.set_ylabel("Amplitude")
        ax1.grid(True, alpha=0.3)
        
        # Spectrogram
        ax2 = fig.add_subplot(gs[0, 1])
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio.squeeze().numpy())), ref=np.max)
        librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sample_rate, ax=ax2)
        ax2.set_title("Audio Spectrogram", fontsize=10, fontweight='bold')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Frequency (Hz)")
        
        # Preprocessed features
        ax3 = fig.add_subplot(gs[0, 2])
        im = ax3.imshow(preprocessed_features.numpy(), aspect='auto', cmap='viridis', 
                       origin='lower', interpolation='nearest')
        ax3.set_title(f"{preprocessing_type.upper()} Features", fontsize=10, fontweight='bold')
        ax3.set_xlabel("Time Steps")
        ax3.set_ylabel("Features")
        plt.colorbar(im, ax=ax3, label="Value")
        
        # Feature statistics
        ax4 = fig.add_subplot(gs[0, 3])
        feature_means = preprocessed_features.mean(dim=1)
        ax4.bar(range(len(feature_means)), feature_means.numpy(), alpha=0.7, color='skyblue')
        ax4.set_title("Feature Statistics", fontsize=10, fontweight='bold')
        ax4.set_xlabel("Feature Channel")
        ax4.set_ylabel("Mean Value")
        ax4.grid(True, alpha=0.3)
        
        # 2. Layer-by-layer spike raster plots (rows 1-3)
        layer_names = [name.replace('_spikes', '') for name in spike_recordings.keys()]
        for i, layer_name in enumerate(layer_names):
            spikes = spike_recordings[f'{layer_name}_spikes']
            
            # Limit neurons for visibility
            n_neurons = min(spikes.shape[1], 30)
            spikes_vis = spikes[:, :n_neurons]
            
            ax = fig.add_subplot(gs[i+1, 0])
            spike_times, neuron_ids = torch.where(spikes_vis > 0)
            if len(spike_times) > 0:
                ax.scatter(spike_times, neuron_ids, s=8, c='red', alpha=0.7)
            ax.set_title(f"{layer_name.title()} - Spike Raster", fontsize=10, fontweight='bold')
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Neuron ID")
            ax.set_ylim(-0.5, n_neurons - 0.5)
            ax.grid(True, alpha=0.3)
            
            # Membrane potential heatmap
            ax = fig.add_subplot(gs[i+1, 1])
            membrane = membrane_recordings[f'{layer_name}_membrane']
            membrane_vis = membrane[:, :n_neurons]
            im = ax.imshow(membrane_vis.T.numpy(), aspect='auto', cmap='RdBu_r', 
                          origin='lower', interpolation='nearest')
            ax.set_title(f"{layer_name.title()} - Membrane Potential", fontsize=10, fontweight='bold')
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Neuron ID")
            plt.colorbar(im, ax=ax, label="Potential")
            
            # Spike rate over time
            ax = fig.add_subplot(gs[i+1, 2])
            spike_rate = spikes.mean(dim=1)
            ax.plot(spike_rate.numpy(), color='red', linewidth=2, alpha=0.8)
            ax.set_title(f"{layer_name.title()} - Spike Rate", fontsize=10, fontweight='bold')
            ax.set_xlabel("Time Steps")
            ax.set_ylabel("Spike Rate")
            ax.grid(True, alpha=0.3)
            
            # Neuron activity distribution
            ax = fig.add_subplot(gs[i+1, 3])
            total_spikes_per_neuron = spikes.sum(dim=0)
            ax.hist(total_spikes_per_neuron.numpy(), bins=15, alpha=0.7, 
                   color='lightcoral', edgecolor='black')
            ax.set_title(f"{layer_name.title()} - Spike Distribution", fontsize=10, fontweight='bold')
            ax.set_xlabel("Total Spikes")
            ax.set_ylabel("Frequency")
            ax.grid(True, alpha=0.3)
        
        # 3. Network-wide statistics (bottom row)
        # Overall spike statistics
        ax_stats1 = fig.add_subplot(gs[4, 0])
        layer_names_clean = [name.replace('_spikes', '') for name in spike_recordings.keys()]
        total_spikes = [spikes.sum().item() for spikes in spike_recordings.values()]
        ax_stats1.bar(layer_names_clean, total_spikes, alpha=0.7, color='red')
        ax_stats1.set_title("Total Spikes per Layer", fontsize=10, fontweight='bold')
        ax_stats1.set_ylabel("Total Spikes")
        ax_stats1.tick_params(axis='x', rotation=45)
        ax_stats1.grid(True, alpha=0.3)
        
        # Average spike rates
        ax_stats2 = fig.add_subplot(gs[4, 1])
        avg_spike_rates = [spikes.mean().item() for spikes in spike_recordings.values()]
        ax_stats2.bar(layer_names_clean, avg_spike_rates, alpha=0.7, color='blue')
        ax_stats2.set_title("Average Spike Rate per Layer", fontsize=10, fontweight='bold')
        ax_stats2.set_ylabel("Average Spike Rate")
        ax_stats2.tick_params(axis='x', rotation=45)
        ax_stats2.grid(True, alpha=0.3)
        
        # Membrane potential evolution
        ax_stats3 = fig.add_subplot(gs[4, 2])
        for layer_name, membrane in membrane_recordings.items():
            clean_name = layer_name.replace('_membrane', '')
            mem_mean = membrane.mean(dim=1)
            ax_stats3.plot(mem_mean.numpy(), label=clean_name, alpha=0.8, linewidth=2)
        ax_stats3.set_title("Membrane Potential Evolution", fontsize=10, fontweight='bold')
        ax_stats3.set_xlabel("Time Steps")
        ax_stats3.set_ylabel("Average Potential")
        ax_stats3.legend()
        ax_stats3.grid(True, alpha=0.3)
        
        # Training history (if available)
        ax_stats4 = fig.add_subplot(gs[4, 3])
        if training_history is not None and 'train_loss' in training_history:
            epochs = range(1, len(training_history['train_loss']) + 1)
            ax_stats4.plot(epochs, training_history['train_loss'], label='Train Loss', 
                          color='blue', linewidth=2)
            if 'val_loss' in training_history:
                ax_stats4.plot(epochs, training_history['val_loss'], label='Val Loss', 
                              color='red', linewidth=2)
            ax_stats4.set_title("Training History", fontsize=10, fontweight='bold')
            ax_stats4.set_xlabel("Epoch")
            ax_stats4.set_ylabel("Loss")
            ax_stats4.legend()
            ax_stats4.grid(True, alpha=0.3)
        else:
            ax_stats4.text(0.5, 0.5, "No training history available", 
                          ha='center', va='center', transform=ax_stats4.transAxes)
            ax_stats4.set_title("Training History", fontsize=10, fontweight='bold')
        
        # 4. Overall network activity heatmap (bottom row, full width)
        ax_heatmap = fig.add_subplot(gs[5, :])
        # Combine all layers for overall activity view
        all_activities = []
        for layer_name, spikes in spike_recordings.items():
            clean_name = layer_name.replace('_spikes', '')
            activity = spikes.mean(dim=0)  # Average over time
            all_activities.append(activity.numpy())
        
        combined_activity = np.concatenate(all_activities)
        activity_heatmap = combined_activity.reshape(1, -1)
        
        im = ax_heatmap.imshow(activity_heatmap, aspect='auto', cmap='viridis', 
                              interpolation='nearest')
        ax_heatmap.set_title("Complete Network Activity Heatmap", fontsize=12, fontweight='bold')
        ax_heatmap.set_xlabel("Neuron ID (All Layers)")
        ax_heatmap.set_ylabel("Activity Level")
        ax_heatmap.set_yticks([])
        
        # Add layer separators
        current_pos = 0
        for i, activity in enumerate(all_activities):
            current_pos += len(activity)
            if i < len(all_activities) - 1:
                ax_heatmap.axvline(current_pos - 0.5, color='white', linewidth=3)
        
        plt.colorbar(im, ax=ax_heatmap, label="Average Spike Rate")
        
        plt.suptitle(title, fontsize=18, fontweight='bold')
        plt.tight_layout()
        plt.show()


def create_visualization_example():
    """
    Create a comprehensive visualization example
    """
    print("Creating comprehensive visualization example...")
    
    # This function would be used in the main example to demonstrate all visualizations
    visualizer = SNNVisualizer()
    
    # Example usage would be:
    # visualizer.plot_audio_preprocessing(audio, preprocessed_features, preprocessing_type="mel")
    # visualizer.plot_lif_neuron_firing(spike_recordings, membrane_recordings)
    # visualizer.plot_layer_activity_summary(spike_recordings, membrane_recordings)
    # visualizer.plot_network_dynamics(spike_recordings, membrane_recordings, training_history)
    # visualizer.plot_comprehensive_analysis(audio, preprocessed_features, spike_recordings, 
    #                                       membrane_recordings, training_history=training_history)
    
    return visualizer 