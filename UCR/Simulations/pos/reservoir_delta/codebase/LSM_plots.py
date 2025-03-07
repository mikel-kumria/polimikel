# LSM_plots.py
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import plotly.graph_objects as go

def animate_reservoir_activity(spike_record, output_path, fps=10):
    """
    Create an animated heatmap (MP4) showing reservoir spiking activity over time.
    Assumes spike_record shape: (time_steps, batch_size, reservoir_size) with reservoir_size=100.
    Reshapes each frame to a 10x10 grid and uses a binary colormap (dark for 0, light for 1).
    """
    time_steps, batch_size, reservoir_size = spike_record.shape
    fig, ax = plt.subplots(figsize=(6, 6), dpi=150)
    
    def update_frame(t):
        ax.clear()
        frame_data = spike_record[t, 0, :].reshape(10, 10)
        im = ax.imshow(frame_data, cmap='binary', vmin=0, vmax=1)
        ax.set_title(f"Time step {t+1}")
        ax.axis('off')
        return im,
    
    ani = animation.FuncAnimation(fig, update_frame, frames=time_steps, interval=1000/fps)
    writer = FFMpegWriter(fps=fps, codec='libx264')
    ani.save(output_path, writer=writer)
    plt.close(fig)

def animate_HPO_heatmap(metric_grid_time, beta_values, threshold_values, output_path, fps=10):
    """
    Create an animated heatmap video (MP4) showing the hyperparameter grid (V_threshold vs beta_reservoir)
    and the average firing rate at each time step.
    metric_grid_time: 3D array of shape (T, n_threshold, n_beta)
    """
    T = metric_grid_time.shape[0]
    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    
    def update_frame(t):
        ax.clear()
        im = ax.imshow(metric_grid_time[t], origin='lower', aspect='auto',
                       extent=[min(beta_values), max(beta_values), min(threshold_values), max(threshold_values)],
                       cmap='viridis')
        ax.set_title(f"Avg Firing Rate at Time Step {t+1}")
        ax.set_xlabel("Beta Reservoir")
        ax.set_ylabel("V_threshold")
        return im,
    
    ani = animation.FuncAnimation(fig, update_frame, frames=T, interval=1000/fps)
    writer = FFMpegWriter(fps=fps, codec='libx264')
    ani.save(output_path, writer=writer)
    plt.close(fig)

def animate_HPO_3D_surface(metric_grid_time, beta_values, threshold_values, output_path, fps=10):
    """
    Create a 3D animated plot (MP4) of average firing rate vs V_threshold vs beta_reservoir over time.
    """
    T = metric_grid_time.shape[0]
    fig = plt.figure(figsize=(10, 8), dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(beta_values, threshold_values)
    
    def update_frame(t):
        ax.clear()
        surf = ax.plot_surface(X, Y, metric_grid_time[t], cmap='viridis', edgecolor='none')
        ax.set_title(f"3D Surface at Time Step {t+1}")
        ax.set_xlabel("Beta Reservoir")
        ax.set_ylabel("V_threshold")
        ax.set_zlabel("Avg Firing Rate")
        return surf,
    
    ani = animation.FuncAnimation(fig, update_frame, frames=T, interval=1000/fps)
    writer = FFMpegWriter(fps=fps, codec='libx264')
    ani.save(output_path, writer=writer)
    plt.close(fig)

def plot_static_heatmap_reach_time(metric_grid_time, beta_values, threshold_values, target_rate, output_path):
    """
    Create a static 2D heatmap showing, for each hyperparameter configuration (V_threshold x beta_reservoir),
    the time step at which the average firing rate first reaches target_rate.
    Configurations that never reach target_rate are colored red.
    metric_grid_time: 3D array of shape (T, n_threshold, n_beta)
    """
    T, n_thresh, n_beta = metric_grid_time.shape
    reach_time = np.full((n_thresh, n_beta), T+1)  # default: never reached
    for i in range(n_thresh):
        for j in range(n_beta):
            times = np.where(metric_grid_time[:, i, j] >= target_rate)[0]
            if times.size > 0:
                reach_time[i, j] = times[0] + 1  # time step number (1-indexed)
    never_mask = (reach_time == T+1)
    
    plt.figure(figsize=(8, 6), dpi=150)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='red')
    masked_data = np.ma.masked_where(never_mask, reach_time)
    plt.imshow(masked_data, origin='lower', aspect='auto',
               extent=[min(beta_values), max(beta_values), min(threshold_values), max(threshold_values)],
               cmap=cmap)
    plt.colorbar(label=f"Time step (1-indexed) to reach rate {target_rate:.2f}")
    plt.xlabel("Beta Reservoir")
    plt.ylabel("V_threshold")
    plt.title(f"Time to reach avg firing rate {target_rate:.2f}")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
