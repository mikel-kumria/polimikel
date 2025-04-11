import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import plotly.graph_objects as go

def load_fr_time(fr_time_file):
    """
    Loads the aggregated FR_time.npy file.
    Returns:
        FR_time: NumPy array of shape (time_steps, n_threshold, n_beta)
    """
    return np.load(fr_time_file)

def plot_all_static_3d_surfaces(fr_time_file, beta_values, threshold_values, spectral_radius, output_folder):
    """
    Loads FR_time from file and loops over all time steps to create and save static 3D surface plots.
    Each image is saved in the "static3Dplots" subfolder.
    """
    # Create the subfolder for 3D static plots.
    static3D_dir = os.path.join(output_folder, "static3Dplots")
    os.makedirs(static3D_dir, exist_ok=True)
    
    FR_time = load_fr_time(fr_time_file)
    T = FR_time.shape[0]
    X, Y = np.meshgrid(beta_values, threshold_values)
    
    for t in range(T):
        FR = FR_time[t]
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=30, azim=120)
        surf = ax.plot_surface(X, Y, FR, cmap='viridis', edgecolor='none')
        ax.set_xlabel('Beta Reservoir')
        ax.set_ylabel('V_threshold')
        ax.set_zlabel('Avg Firing Rate')
        ax.set_title(f"3D Surface at Time Step {t} ($\\rho$: {spectral_radius:.2f})")
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.tight_layout()
        save_path = os.path.join(static3D_dir, f"3d_surface_t{t}.png")
        plt.savefig(save_path, dpi=600)
        plt.close()

def plot_all_static_2d_heatmaps(fr_time_file, beta_values, threshold_values, spectral_radius, output_folder):
    """
    Loads FR_time from file and loops over all time steps to create and save static 2D heatmaps.
    Each image is saved in the "static2Dplots" subfolder.
    """
    static2D_dir = os.path.join(output_folder, "static2Dplots")
    os.makedirs(static2D_dir, exist_ok=True)
    
    FR_time = load_fr_time(fr_time_file)
    T = FR_time.shape[0]
    
    for t in range(T):
        FR = FR_time[t]
        plt.figure(figsize=(8, 6))
        plt.imshow(FR, origin='lower', aspect='auto',
                   extent=[min(beta_values), max(beta_values), min(threshold_values), max(threshold_values)],
                   cmap='viridis')
        plt.colorbar(label='Avg Firing Rate')
        plt.xlabel('Beta Reservoir')
        plt.ylabel('V_threshold')
        plt.title(f"2D Heatmap at Time Step {t} ($\\rho$: {spectral_radius:.2f})")
        plt.tight_layout()
        save_path = os.path.join(static2D_dir, f"2d_heatmap_t{t}.png")
        plt.savefig(save_path, dpi=600)
        plt.close()

def plot_interactive_animated_3d_from_file(fr_time_file, beta_values, threshold_values, spectral_radius, output_folder):
    """
    Loads FR_time from file and creates an interactive animated 3D surface plot over time,
    with a slider for manual frame control. Exports the animation as an HTML file.
    """
    FR_time = load_fr_time(fr_time_file)
    T, _, _ = FR_time.shape
    
    # Create frames for each time step.
    frames = [go.Frame(data=[go.Surface(
                    z=FR_time[t], 
                    x=beta_values, 
                    y=threshold_values, 
                    colorscale='Viridis'
                )], name=str(t))
              for t in range(T)]
    
    # Create slider steps so that you can select a frame manually.
    slider_steps = []
    for t in range(T):
        step = dict(
            method='animate',
            args=[[str(t)],
                  {'frame': {'duration': 0, 'redraw': True},
                   'mode': 'immediate'}],
            label=str(t)
        )
        slider_steps.append(step)

    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Frame: "},
        pad={"t": 50},
        steps=slider_steps
    )]

    # Build the initial surface.
    init_surface = go.Surface(
        z=FR_time[0], 
        x=beta_values, 
        y=threshold_values, 
        colorscale='Viridis'
    )
    
    # Create the figure and add frames and slider.
    fig = go.Figure(data=[init_surface], frames=frames)
    fig.update_layout(
        title=f"Animated 3D Surface of Avg Firing Rate (ρ: {spectral_radius:.2f})",
        scene=dict(
            xaxis_title="Beta Reservoir",
            yaxis_title="V_threshold",
            zaxis_title="Avg Firing Rate"
        ),
        sliders=sliders,
        updatemenus=[{
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}],
                    "label": "Play",
                    "method": "animate",
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
                    "label": "Pause",
                    "method": "animate",
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "showactive": False,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    )

    # Save the interactive figure as an HTML file.
    save_path = os.path.join(output_folder, "interactive_3d_animated.html")
    fig.write_html(save_path)

def plot_interactive_animated_2d_from_file(fr_time_file, beta_values, threshold_values, spectral_radius, output_folder):
    """
    Loads FR_time from file and creates an interactive animated 2D heatmap over time.
    Adds a slider to manually control the frame being displayed.
    Exports the animation as an HTML file.
    """
    # Load the FR_time data.
    FR_time = load_fr_time(fr_time_file)
    T, _, _ = FR_time.shape

    # Build the animation frames and slider steps.
    frames = []
    slider_steps = []
    for t in range(T):
        frame = go.Frame(
            data=[go.Heatmap(z=FR_time[t],
                             x=beta_values,
                             y=threshold_values,
                             colorscale='Viridis')],
            name=str(t)
        )
        frames.append(frame)

        # Create a slider step for this frame.
        step = {
            "args": [
                [str(t)],
                {"frame": {"duration": 0, "redraw": True},
                 "mode": "immediate"}
            ],
            "label": str(t),
            "method": "animate"
        }
        slider_steps.append(step)

    # Create the initial heatmap.
    init_heat = go.Heatmap(
        z=FR_time[0],
        x=beta_values,
        y=threshold_values,
        colorscale='Viridis'
    )

    # Build the figure.
    fig = go.Figure(data=[init_heat], frames=frames)
    fig.update_layout(
        title=f"Animated 2D Heatmap of Avg Firing Rate (ρ: {spectral_radius:.2f})",
        xaxis_title="Beta Reservoir",
        yaxis_title="V_threshold",
        sliders=[{
            "active": 0,
            "currentvalue": {"prefix": "Frame: "},
            "pad": {"t": 50},
            "steps": slider_steps
        }],
        updatemenus=[{
            "type": "buttons",
            "buttons": [
                {
                    "label": "Play",
                    "method": "animate",
                    "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]
                },
                {
                    "label": "Pause",
                    "method": "animate",
                    "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]
                }
            ],
            "direction": "left",
            "pad": {"r": 10, "t": 70},
            "showactive": False,
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }]
    )

    # Save the interactive figure as an HTML file.
    save_path = os.path.join(output_folder, "interactive_2d_animated.html")
    fig.write_html(save_path)