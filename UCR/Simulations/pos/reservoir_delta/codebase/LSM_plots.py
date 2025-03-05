import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # for 3D plotting
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter
import plotly.graph_objects as go
import pandas as pd


def plot_static_3d_surface(firing_rate_grid, beta_values, threshold_values, spectral_radius, output_folder, writer):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=120)
    X, Y = np.meshgrid(beta_values, threshold_values)
    surf = ax.plot_surface(X, Y, firing_rate_grid, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Beta Reservoir')
    ax.set_ylabel('Threshold')
    ax.set_zlabel('Avg Firing Rate')
    # Use a raw string to allow LaTeX rendering
    ax.set_title(r'3D Surface Plot of Avg Firing Rate: $\rho$: ' + f'{spectral_radius:.2f}')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    surface_path = os.path.join(output_folder, '3d_surface_plot.png')
    plt.savefig(surface_path, dpi=600)
    writer.add_figure("3D_Surface_Plot", fig)
    plt.close()

def plot_static_heatmap(firing_rate_grid, beta_values, threshold_values, spectral_radius, output_folder, writer):
    plt.figure(figsize=(8, 6))
    plt.imshow(firing_rate_grid, origin='lower', aspect='auto',
               extent=[min(beta_values), max(beta_values), min(threshold_values), max(threshold_values)],
               cmap='viridis')
    plt.colorbar(label='Avg Firing Rate')
    plt.xlabel('Beta Reservoir')
    plt.ylabel('Threshold')
    plt.title(r'Avg Firing Rate: $\rho$: ' + f'{spectral_radius:.2f}')
    plt.tight_layout()
    heatmap_path = os.path.join(output_folder, 'heatmap_firing_rate.png')
    plt.savefig(heatmap_path, dpi=600)
    writer.add_figure("Heatmap_Firing_Rate", plt.gcf())
    plt.close()

# def plot_static_spike_raster(spike_record, output_folder, writer):
#     plt.figure(figsize=(10, 6))
#     for neuron in range(spike_record.shape[1]):
#         spike_times = np.where(spike_record[:, neuron] > 0)[0]
#         plt.scatter(spike_times, np.full_like(spike_times, neuron), s=10)
#     plt.xlabel('Time Step')
#     plt.ylabel('Neuron Index')
#     plt.title('Spike Raster Plot')
#     plt.yticks(range(spike_record.shape[1]))
#     plt.grid(True)
#     plt.tight_layout()
#     spike_raster_path = os.path.join(output_folder, 'spike_raster_plot.png')
#     plt.savefig(spike_raster_path, dpi=600)
#     writer.add_figure("Spike_Raster_Plot", plt.gcf())
#     plt.close()

# def plot_static_membrane_traces(mem_record, output_folder, writer):
#     plt.figure(figsize=(10, 6))
#     for neuron in range(mem_record.shape[1]):
#         plt.plot(mem_record[:, neuron], label=f'Neuron {neuron}')
#     plt.xlabel('Time Step')
#     plt.ylabel('Membrane Potential')
#     plt.title('Membrane Potential Traces')
#     plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
#     plt.grid(True)
#     plt.tight_layout()
#     mem_trace_path = os.path.join(output_folder, 'membrane_potential_traces.png')
#     plt.savefig(mem_trace_path, dpi=600)
#     writer.add_figure("Membrane_Potential_Traces", plt.gcf())
#     plt.close()

# def plot_static_isi_histogram(all_intervals, output_folder, writer):
#     plt.figure(figsize=(8, 6))
#     plt.hist(all_intervals, bins=30, color='skyblue', edgecolor='black')
#     plt.xlabel('Inter-Spike Interval (Time Steps)')
#     plt.ylabel('Count')
#     plt.title('Inter-Spike Interval Histogram')
#     plt.grid(True)
#     plt.tight_layout()
#     isi_path = os.path.join(output_folder, 'isi_histogram.png')
#     plt.savefig(isi_path, dpi=600)
#     writer.add_figure("ISI_Histogram", plt.gcf())
#     plt.close()

def plot_static_weight_matrix(weights, output_folder, writer):
    plt.figure(figsize=(6, 5))
    plt.imshow(weights, cmap='inferno', aspect='auto')
    plt.colorbar(label='Weight Value')
    plt.xlabel('Post-synaptic Neuron')
    plt.ylabel('Pre-synaptic Neuron')
    plt.title('Recurrent Weight Matrix Heatmap')
    plt.tight_layout()
    weight_heatmap_path = os.path.join(output_folder, 'weight_matrix_heatmap.png')
    plt.savefig(weight_heatmap_path, dpi=600)
    writer.add_figure("Weight_Matrix_Heatmap", plt.gcf())
    plt.close()

# def plot_static_eigenvalues(eigenvalues, output_folder, writer):
#     plt.figure(figsize=(8, 6))
#     plt.scatter(eigenvalues.real, eigenvalues.imag, c='purple', edgecolors='k')
#     plt.xlabel('Real Part')
#     plt.ylabel('Imaginary Part')
#     plt.title('Eigenvalues of the Recurrent Weight Matrix')
#     plt.grid(True)
#     plt.axhline(0, color='black', linewidth=0.5)
#     plt.axvline(0, color='black', linewidth=0.5)
#     plt.tight_layout()
#     eigen_path = os.path.join(output_folder, 'eigenvalues.png')
#     plt.savefig(eigen_path, dpi=600)
#     writer.add_figure("Eigenvalues", plt.gcf())
#     plt.close()

def plot_interactive_ts_input(tsv_file, output_folder):
    data = pd.read_csv(tsv_file, sep='\t', header=0)
    x_sample = data.iloc[0, 1:].values.astype(np.float32)
    num_steps = len(x_sample)
    fig_input = go.Figure(data=go.Scatter(x=list(range(num_steps)), y=x_sample, mode='lines+markers'))
    fig_input.update_layout(title='TSV Input Time Series',
                            xaxis_title='Time Step',
                            yaxis_title='Value')
    fig_input.write_html(os.path.join(output_folder, 'tsv_input_interactive.html'))

def plot_interactive_3d_surface(firing_rate_grid, beta_values, threshold_values, spectral_radius, output_folder):
    # Use a raw string for LaTeX formatting.
    fig_3d = go.Figure(data=[go.Surface(z=firing_rate_grid, x=beta_values, y=threshold_values, colorscale='Viridis')])
    fig_3d.update_layout(title=r'3D Surface Plot of Avg Firing Rate: $\rho$: ' + f'{spectral_radius:.2f}',
                         scene=dict(xaxis_title='Beta Reservoir',
                                    yaxis_title='Threshold',
                                    zaxis_title='Avg Firing Rate'))
    fig_3d.write_html(os.path.join(output_folder, '3d_surface_plot_interactive.html'))

def plot_interactive_heatmap(firing_rate_grid, beta_values, threshold_values, spectral_radius, output_folder):
    fig_heatmap = go.Figure(data=go.Heatmap(z=firing_rate_grid, x=beta_values, y=threshold_values, colorscale='Viridis'))
    fig_heatmap.update_layout(title=r'Heatmap of Avg Firing Rate: $\rho$: ' + f'{spectral_radius:.2f}',
                              xaxis_title='Beta Reservoir',
                              yaxis_title='Threshold')
    fig_heatmap.write_html(os.path.join(output_folder, 'heatmap_firing_rate_interactive.html'))

# def plot_interactive_spike_raster(spike_record, output_folder):
#     fig_raster = go.Figure()
#     for neuron in range(spike_record.shape[1]):
#         spike_times = np.where(spike_record[:, neuron] > 0)[0]
#         fig_raster.add_trace(go.Scatter(x=spike_times,
#                                         y=np.full_like(spike_times, neuron),
#                                         mode='markers',
#                                         name=f'Neuron {neuron}',
#                                         marker=dict(size=6)))
#     fig_raster.update_layout(title='Spike Raster Plot',
#                              xaxis_title='Time Step',
#                              yaxis_title='Neuron Index')
#     fig_raster.write_html(os.path.join(output_folder, 'spike_raster_plot_interactive.html'))

# def plot_interactive_membrane_traces(mem_record, output_folder):
#     time_steps = mem_record.shape[0]
#     time_axis = list(range(time_steps))
#     fig_mem = go.Figure()
#     for neuron in range(mem_record.shape[1]):
#         fig_mem.add_trace(go.Scatter(x=time_axis, y=mem_record[:, neuron],
#                                      mode='lines',
#                                      name=f'Neuron {neuron}'))
#     fig_mem.update_layout(title='Membrane Potential Traces',
#                           xaxis_title='Time Step',
#                           yaxis_title='Membrane Potential')
#     fig_mem.write_html(os.path.join(output_folder, 'membrane_potential_traces_interactive.html'))

# def plot_interactive_isi_histogram(all_intervals, output_folder):
#     fig_isi = go.Figure(data=[go.Histogram(x=all_intervals, nbinsx=30, marker_color='skyblue')])
#     fig_isi.update_layout(title='Inter-Spike Interval Histogram',
#                           xaxis_title='Inter-Spike Interval (Time Steps)',
#                           yaxis_title='Count')
#     fig_isi.write_html(os.path.join(output_folder, 'isi_histogram_interactive.html'))

# def plot_interactive_weight_matrix(weights, output_folder):
#     fig_weights = go.Figure(data=go.Heatmap(z=weights, colorscale='RdBu'))
#     fig_weights.update_layout(title='Recurrent Weight Matrix Heatmap',
#                               xaxis_title='Post-synaptic Neuron',
#                               yaxis_title='Pre-synaptic Neuron')
#     fig_weights.write_html(os.path.join(output_folder, 'weight_matrix_heatmap_interactive.html'))

# def plot_interactive_eigenvalues(eigenvalues, output_folder):
#     fig_eigen = go.Figure(data=go.Scatter(x=eigenvalues.real,
#                                           y=eigenvalues.imag,
#                                           mode='markers',
#                                           marker=dict(size=10, color='purple')))
#     fig_eigen.update_layout(title='Eigenvalues of the Recurrent Weight Matrix',
#                             xaxis_title='Real Part',
#                             yaxis_title='Imaginary Part')
#     fig_eigen.write_html(os.path.join(output_folder, 'eigenvalues_interactive.html'))

def plot_interactive_animated_3d_surface(FR_time, beta_values, threshold_values, spectral_radius, output_folder):
    """
    Creates an animated 3D surface plot where FR_time is a 3D array of shape
    (T, n_threshold, n_beta) representing the firing rate at each time step.
    The z-axis is fixed to the range [0, 1] regardless of the data.
    Also exports the animation as an HTML file.
    """
    T, n_thresh, n_beta = FR_time.shape
    frames = []
    for t in range(T):
        frame = go.Frame(
            data=[go.Surface(z=FR_time[t], x=beta_values, y=threshold_values, colorscale='Viridis')],
            name=str(t),
            layout=dict(
                scene=dict(
                    zaxis=dict(range=[0, 1], autorange=False)
                )
            )
        )
        frames.append(frame)
    initial_data = go.Surface(z=FR_time[0], x=beta_values, y=threshold_values, colorscale='Viridis')
    fig = go.Figure(data=[initial_data], frames=frames)
    fig.update_layout(
        title=r"Animated 3D Surface of Firing Rate over Time: $\rho$: " + f"{spectral_radius:.2f}",
        scene=dict(
            xaxis_title="Beta Reservoir",
            yaxis_title="Threshold",
            zaxis=dict(title="Firing Rate", range=[0, 1], autorange=False)
        ),
        updatemenus=[dict(
            type="buttons",
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate", "transition": {"duration": 0}}])
            ]
        )]
    )
    fig.write_html(os.path.join(output_folder, "animated_3d_surface.html"))


def plot_interactive_animated_heatmap(FR_time, beta_values, threshold_values, spectral_radius, output_folder):
    """
    Creates an animated heatmap where FR_time is a 3D array of shape
    (T, n_threshold, n_beta) representing the firing rate at each time step.
    Also exports the animation as an HTML file.
    """
    T, n_thresh, n_beta = FR_time.shape
    frames = []
    for t in range(T):
        frame = go.Frame(data=[go.Heatmap(z=FR_time[t], x=beta_values, y=threshold_values, colorscale='Viridis')],
                         name=str(t))
        frames.append(frame)
    initial_data = go.Heatmap(z=FR_time[0], x=beta_values, y=threshold_values, colorscale='Viridis')
    fig = go.Figure(data=[initial_data], frames=frames)
    fig.update_layout(
        title=r"Animated Heatmap of Firing Rate over Time: $\rho$: " + f"{spectral_radius:.2f}",
        xaxis_title="Beta Reservoir",
        yaxis_title="Threshold",
        updatemenus=[dict(
            type="buttons",
            buttons=[dict(label="Play",
                          method="animate",
                          args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]),
                     dict(label="Pause",
                          method="animate",
                          args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])]
        )]
    )
    fig.write_html(os.path.join(output_folder, "animated_heatmap.html"))

def animate_3d_video(FR_time, beta_values, threshold_values, spectral_radius, output_folder, fps=10):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=120)
    X, Y = np.meshgrid(beta_values, threshold_values)
    
    def update_frame(t):
        ax.clear()
        surf = ax.plot_surface(X, Y, FR_time[t], cmap='viridis', edgecolor='none')
        ax.set_xlabel('Beta Reservoir')
        ax.set_ylabel('Threshold')
        ax.set_zlabel('Firing Rate')
        ax.set_title(r"Animated 3D Surface: $\rho$: " + f"{spectral_radius:.2f}, Time: {t}")
        return surf,
    
    ani = animation.FuncAnimation(fig, update_frame, frames=FR_time.shape[0], interval=1000/fps)
    video_path = os.path.join(output_folder, "animated_3d_surface.mp4")
    #ani.save(video_path, fps=fps, extra_args=['-vcodec', 'libx264'])
    writer = FFMpegWriter(fps=fps, codec='libx264')
    ani.save(video_path, writer=writer)
    plt.close(fig)

def animate_heatmap_video(FR_time, beta_values, threshold_values, spectral_radius, output_folder, fps=10):
    fig, ax = plt.subplots(figsize=(8, 6))
    
    def update_frame(t):
        ax.clear()
        im = ax.imshow(
            FR_time[t],
            origin='lower',
            aspect='auto',
            extent=[min(beta_values), max(beta_values), min(threshold_values), max(threshold_values)],
            cmap='viridis'
        )
        ax.set_xlabel('Beta Reservoir')
        ax.set_ylabel('Threshold')
        ax.set_title(r"Animated Heatmap: $\rho$: " + f"{spectral_radius:.2f}, Time: {t}")
        return [im]
    
    # Create and keep the animation object in a variable
    anim = animation.FuncAnimation(fig, update_frame, frames=FR_time.shape[0], interval=1000/fps)
    video_path = os.path.join(output_folder, "animated_heatmap.mp4")
    
    # Save the animation
    #anim.save(video_path, fps=fps, extra_args=['-vcodec', 'libx264'])
    writer = FFMpegWriter(fps=fps, codec='libx264')
    anim.save(video_path, writer=writer)
    plt.close(fig)


# def plot_interactive_animated_3d_membrane(MP_time, beta_values, threshold_values, spectral_radius, output_folder):
#     """
#     Creates an interactive animated 3D surface plot for membrane potential.
#     MP_time is a 3D array of shape (T, n_threshold, n_beta) where each frame corresponds to the average membrane potential at that time step.
#     The threshold axis is reversed so that high values appear at the bottom.
#     """
#     T, n_thresh, n_beta = MP_time.shape

#     frames = []
#     for t in range(T):
#         frame = go.Frame(
#             data=[go.Surface(z=MP_time[t], x=beta_values, y=threshold_values, colorscale='Viridis')],
#             name=str(t)
#         )
#         frames.append(frame)
        
#     initial_data = go.Surface(z=MP_time[0], x=beta_values, y=threshold_values, colorscale='Viridis')
#     fig = go.Figure(data=[initial_data], frames=frames)
#     fig.update_layout(
#         title=r"Animated 3D Membrane Potential over Time: $\rho$: " + f"{spectral_radius:.2f}",
#         scene=dict(xaxis_title="Beta Reservoir",
#                    yaxis_title="Threshold",
#                    zaxis_title="Membrane Potential"),
#         updatemenus=[dict(
#             type="buttons",
#             buttons=[
#                 dict(label="Play",
#                      method="animate",
#                      args=[None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]),
#                 dict(label="Pause",
#                      method="animate",
#                      args=[[None], {"frame": {"duration": 0, "redraw": False},
#                                     "mode": "immediate", "transition": {"duration": 0}}])
#             ]
#         )]
#     )
#     fig.write_html(os.path.join(output_folder, "animated_3d_membrane.html"))

def plot_static_3d_surface_classification(metric_grid, beta_values, threshold_values, output_folder, writer):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(elev=30, azim=120)
    X, Y = np.meshgrid(beta_values, threshold_values)
    surf = ax.plot_surface(X, Y, metric_grid, cmap='viridis', edgecolor='none')
    ax.set_xlabel('Beta Reservoir')
    ax.set_ylabel('Threshold')
    ax.set_zlabel('F1 Score (Abnormal Class)')
    ax.set_title('3D Surface of F1 Score for Abnormal Class')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    surface_path = os.path.join(output_folder, '3d_surface_classification.png')
    plt.savefig(surface_path, dpi=600)
    writer.add_figure("3D_Surface_Classification", fig)
    plt.close()

def plot_static_heatmap_classification(metric_grid, beta_values, threshold_values, output_folder, writer):
    plt.figure(figsize=(8, 6))
    plt.imshow(metric_grid, origin='lower', aspect='auto',
               extent=[min(beta_values), max(beta_values), min(threshold_values), max(threshold_values)],
               cmap='viridis')
    plt.colorbar(label='F1 Score (Abnormal Class)')
    plt.xlabel('Beta Reservoir')
    plt.ylabel('Threshold')
    plt.title('Heatmap of F1 Score for Abnormal Class')
    plt.tight_layout()
    heatmap_path = os.path.join(output_folder, 'heatmap_classification.png')
    plt.savefig(heatmap_path, dpi=600)
    writer.add_figure("Heatmap_Classification", plt.gcf())
    plt.close()