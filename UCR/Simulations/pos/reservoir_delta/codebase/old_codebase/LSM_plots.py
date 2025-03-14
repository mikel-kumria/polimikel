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

# --- New Plotting Functions ---

def plot_spike_record_static(spike_record, output_path):
    """
    Plots the spike_record for each neuron over time on a static matplotlib figure.
    Expects spike_record with shape: (time_steps, batch_size, reservoir_size) (typically batch_size=1).
    """
    time_steps = spike_record.shape[0]
    num_neurons = spike_record.shape[2]
    plt.figure(figsize=(10, 6))
    for i in range(num_neurons):
        plt.plot(np.arange(time_steps), spike_record[:, 0, i], label=f"Neuron {i+1}", alpha=0.8)
    plt.xlabel("Time step")
    plt.ylabel("Spike")
    plt.title("Spike Record per Neuron")
    # Optionally, if there are many neurons the legend might be crowded.
    if num_neurons <= 10:
        plt.legend(loc='upper right', fontsize='small')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

def plot_spike_record_interactive(spike_record, output_path):
    """
    Creates an interactive plot of spike_record for each neuron using Plotly.
    The plot is saved as an HTML file.
    """
    time_steps = spike_record.shape[0]
    num_neurons = spike_record.shape[2]
    fig = go.Figure()
    for i in range(num_neurons):
        fig.add_trace(go.Scatter(
            x=list(range(time_steps)),
            y=spike_record[:, 0, i],
            mode='lines',
            name=f"Neuron {i+1}"
        ))
    fig.update_layout(
        title="Spike Record per Neuron",
        xaxis_title="Time step",
        yaxis_title="Spike"
    )
    fig.write_html(output_path)

def plot_mem_records_all_runs(mem_records_list, output_path):
    """
    Create a grid of subplots showing the membrane potential (mem_record) for each neuron,
    for each HPO run.
    
    mem_records_list: list of tuples (V_threshold, beta_reservoir, mem_record)
      where mem_record has shape: (time_steps, batch_size, reservoir_size)
    
    The grid is arranged with rows corresponding to V_threshold values and columns to beta_reservoir values.
    """
    # Get unique hyperparameter values and sort them.
    v_thresh_set = sorted(set([item[0] for item in mem_records_list]))
    beta_set = sorted(set([item[1] for item in mem_records_list]))
    n_rows = len(v_thresh_set)
    n_cols = len(beta_set)
    
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3), sharex=True, sharey=True)
    # Ensure axs is 2D.
    if n_rows == 1 and n_cols == 1:
        axs = np.array([[axs]])
    elif n_rows == 1 or n_cols == 1:
        axs = np.atleast_2d(axs)
    
    # Create a lookup dict for mem_record by (V_threshold, beta_reservoir)
    mem_dict = {(v, b): mem for (v, b, mem) in mem_records_list}
    
    for i, v in enumerate(v_thresh_set):
        for j, b in enumerate(beta_set):
            ax = axs[i, j]
            mem_record = mem_dict.get((v, b))
            if mem_record is None:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
                continue
            time_steps = mem_record.shape[0]
            num_neurons = mem_record.shape[2]
            for neuron in range(num_neurons):
                ax.plot(np.arange(time_steps), mem_record[:, 0, neuron], alpha=0.7)
            ax.set_title(f"V_thresh={v}, beta={b}")
            if i == n_rows - 1:
                ax.set_xlabel("Time step")
            if j == 0:
                ax.set_ylabel("Membrane Potential")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
