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

def plot_all_static_3d_surfaces(fr_time_file, beta_values, spectral_radius_values, output_folder):
    """
    Loads FR_time from file and loops over all time steps to create and save static 3D surface plots.
    Each image is saved in the "static3Dplots" subfolder. Also creates an animation video.
    """
    # Create the subfolder for 3D static plots.
    static3D_dir = os.path.join(output_folder, "static3Dplots")
    os.makedirs(static3D_dir, exist_ok=True)
    
    FR_time = load_fr_time(fr_time_file)
    T = FR_time.shape[0]
    X, Y = np.meshgrid(beta_values, spectral_radius_values)
    
    for t in range(T):
        FR = FR_time[t]
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(elev=30, azim=120)
        surf = ax.plot_surface(X, Y, FR, cmap='viridis', edgecolor='none')
        ax.set_xlabel('Beta Reservoir')
        ax.set_ylabel('Spectral Radius')
        ax.set_zlabel('Avg Firing Rate')
        ax.set_title(f"3D Surface at Time Step {t}")
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.tight_layout()
        save_path = os.path.join(static3D_dir, f"3d_surface_t{t}.png")
        plt.savefig(save_path, dpi=600)
        plt.close()
    
    # Create animations after generating static plots
    create_animations_from_static_plots(fr_time_file, beta_values, spectral_radius_values, output_folder)

def plot_all_static_2d_heatmaps(fr_time_file, beta_values, spectral_radius_values, output_folder):
    """
    Loads FR_time from file and loops over all time steps to create and save static 2D heatmaps.
    Each image is saved in the "static2Dplots" subfolder. Also creates an animation video.
    """
    static2D_dir = os.path.join(output_folder, "static2Dplots")
    os.makedirs(static2D_dir, exist_ok=True)
    
    FR_time = load_fr_time(fr_time_file)
    T = FR_time.shape[0]
    
    for t in range(T):
        FR = FR_time[t]
        plt.figure(figsize=(8, 6))
        plt.imshow(FR, origin='lower', aspect='auto',
                   extent=[min(beta_values), max(beta_values), min(spectral_radius_values), max(spectral_radius_values)],
                   cmap='viridis')
        plt.colorbar(label='Avg Firing Rate')
        plt.xlabel('Beta Reservoir')
        plt.ylabel('Spectral Radius')
        plt.title(f"2D Heatmap at Time Step {t}")
        plt.tight_layout()
        save_path = os.path.join(static2D_dir, f"2d_heatmap_t{t}.png")
        plt.savefig(save_path, dpi=600)
        plt.close()
    
    # Create animations after generating static plots
    create_animations_from_static_plots(fr_time_file, beta_values, spectral_radius_values, output_folder)

def plot_interactive_animated_3d_from_file(fr_time_file, beta_values, spectral_radius_values, output_folder):
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
                    y=spectral_radius_values, 
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
        y=spectral_radius_values, 
        colorscale='Viridis'
    )
    
    # Create the figure and add frames and slider.
    fig = go.Figure(data=[init_surface], frames=frames)
    fig.update_layout(
        title="Animated 3D Surface of Avg Firing Rate",
        scene=dict(
            xaxis_title="Beta Reservoir",
            yaxis_title="Spectral Radius",
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
            "y": 1.1,
            "yanchor": "top"
        }]
    )

    # Save the interactive figure as an HTML file.
    save_path = os.path.join(output_folder, "interactive_3d_animated.html")
    fig.write_html(save_path)

def plot_interactive_animated_2d_from_file(fr_time_file, beta_values, spectral_radius_values, output_folder):
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
                             y=spectral_radius_values,
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
        y=spectral_radius_values,
        colorscale='Viridis'
    )

    # Build the figure.
    fig = go.Figure(data=[init_heat], frames=frames)
    fig.update_layout(
        title="Animated 2D Heatmap of Avg Firing Rate",
        xaxis_title="Beta Reservoir",
        yaxis_title="Spectral Radius",
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
            "y": 1.1,
            "yanchor": "top"
        }]
    )

    # Save the interactive figure as an HTML file.
    save_path = os.path.join(output_folder, "interactive_2d_animated.html")
    fig.write_html(save_path)

def plot_interactive_convergence_time_heatmap(fr_time_file, beta_values, spectral_radius_values, output_folder):
    """
    Creates an interactive heatmap showing convergence times with hover information.
    """
    FR_time = load_fr_time(fr_time_file)
    T, n_rho, n_beta = FR_time.shape
    
    # Initialize convergence time matrix
    convergence_times = np.full((n_rho, n_beta), T)  # Default to max time steps
    
    # For each configuration, find when firing rate goes below 1%
    for i in range(n_rho):
        for j in range(n_beta):
            firing_rates = FR_time[:, i, j]
            # Find first time step where firing rate < 1%
            below_threshold = np.where(firing_rates < 0.01)[0]
            if len(below_threshold) > 0:
                convergence_times[i, j] = below_threshold[0]
    
    # Create hover text matrix
    hover_text = []
    for i in range(n_rho):
        hover_row = []
        for j in range(n_beta):
            if convergence_times[i, j] >= T:
                hover_row.append(
                    f"β: {beta_values[j]:.3f}<br>" +
                    f"ρ: {spectral_radius_values[i]:.3f}<br>" +
                    f"Status: No convergence in {T} steps"
                )
            else:
                hover_row.append(
                    f"β: {beta_values[j]:.3f}<br>" +
                    f"ρ: {spectral_radius_values[i]:.3f}<br>" +
                    f"Convergence time: {int(convergence_times[i, j])} steps"
                )
        hover_text.append(hover_row)
    
    # Create the heatmap using plotly with Viridis color palette
    fig = go.Figure(data=go.Heatmap(
        z=convergence_times,
        x=beta_values,
        y=spectral_radius_values,
        colorscale='Viridis',
        text=hover_text,
        hoverinfo='text',
        colorbar=dict(
            title='Time steps to <1% firing rate',
            tickmode='array',
            ticktext=[f'≤{T//4}', f'≤{T//2}', f'≤{3*T//4}', f'No convergence'],
            tickvals=[T//4, T//2, 3*T//4, T]
        )
    ))
    
    # Update layout
    fig.update_layout(
        title="Interactive Convergence Time Heatmap",
        xaxis_title="Beta Reservoir",
        yaxis_title="Spectral Radius",
        width=800,
        height=600
    )
    
    # Save the interactive plot
    save_path = os.path.join(output_folder, "convergence_time_heatmap_interactive.html")
    fig.write_html(save_path)

def plot_convergence_time_heatmap(fr_time_file, beta_values, spectral_radius_values, output_folder):
    """
    Creates both static and interactive heatmaps showing how many time steps it takes 
    for each configuration (spectral_radius, beta_reservoir) to reach an average firing 
    rate below 1%. If a configuration never reaches this state, it's marked with 
    a special value.
    """
    FR_time = load_fr_time(fr_time_file)
    T, n_rho, n_beta = FR_time.shape
    
    # Initialize convergence time matrix
    convergence_times = np.full((n_rho, n_beta), T)  # Default to max time steps
    
    # For each configuration, find when firing rate goes below 1%
    for i in range(n_rho):
        for j in range(n_beta):
            firing_rates = FR_time[:, i, j]
            # Find first time step where firing rate < 1%
            below_threshold = np.where(firing_rates < 0.01)[0]
            if len(below_threshold) > 0:
                convergence_times[i, j] = below_threshold[0]
    
    # Create the static plot with extra space on the right for the colorbar
    fig = plt.figure(figsize=(12, 8))
    
    # Create a masked array for converging cases
    converging = convergence_times < T
    masked_convergence = np.ma.masked_where(~converging, convergence_times)
    
    # Create a masked array for non-converging cases
    non_converging = convergence_times >= T
    masked_non_converging = np.ma.masked_where(~non_converging, convergence_times)
    
    # Plot converging cases with viridis colormap
    im1 = plt.imshow(masked_convergence, origin='lower', aspect='auto',
                    extent=[min(beta_values), max(beta_values), min(spectral_radius_values), max(spectral_radius_values)],
                    cmap='viridis')
    
    # Plot non-converging cases in black
    plt.imshow(masked_non_converging, origin='lower', aspect='auto',
               extent=[min(beta_values), max(beta_values), min(spectral_radius_values), max(spectral_radius_values)],
               cmap=plt.cm.colors.ListedColormap(['black']))
    
    # Add colorbar for convergence times
    cbar = plt.colorbar(im1, label='Time steps to <1% firing rate')
    
    # Add text annotation for non-converging cases
    plt.figtext(0.99, 0.02, f'Black = no convergence in {T} time steps',
                horizontalalignment='right', fontsize=9)
    
    plt.xlabel('Beta Reservoir')
    plt.ylabel('Spectral Radius')
    plt.title("Convergence Time Heatmap")
    
    # Adjust layout to prevent overlapping
    plt.tight_layout()
    
    # Save the static plot
    save_path = os.path.join(output_folder, "convergence_time_heatmap.png")
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.close(fig)
    
    # Create and save the interactive version
    plot_interactive_convergence_time_heatmap(fr_time_file, beta_values, spectral_radius_values, output_folder)

def create_animations_from_static_plots(fr_time_file, beta_values, spectral_radius_values, output_folder):
    """
    Creates video animations by combining the existing static plot images.
    Uses the previously generated PNG files in static2Dplots and static3Dplots folders.
    """
    # Create the video subfolder
    video_dir = os.path.join(output_folder, "videos")
    os.makedirs(video_dir, exist_ok=True)
    
    # Get the number of time steps
    FR_time = load_fr_time(fr_time_file)
    T = FR_time.shape[0]
    
    # Create animation from 2D static images
    try:
        # Import imageio for creating videos from images
        import imageio
        
        # Path to 2D static images
        static2D_dir = os.path.join(output_folder, "static2Dplots")
        if os.path.exists(static2D_dir):
            # Get all 2D heatmap PNG files and sort them by time step
            image_files_2d = sorted(
                [os.path.join(static2D_dir, f) for f in os.listdir(static2D_dir) if f.startswith('2d_heatmap_t') and f.endswith('.png')],
                key=lambda x: int(os.path.basename(x).split('t')[1].split('.')[0])  # Extract time step number for sorting
            )
            
            if image_files_2d:
                # Create output video file path
                output_video_2d = os.path.join(video_dir, "2d_heatmap_animation.mp4")
                
                # Load images and create video
                with imageio.get_writer(output_video_2d, fps=4) as writer:
                    for image_file in image_files_2d:
                        image = imageio.imread(image_file)
                        writer.append_data(image)
                
                print(f"Created 2D animation from {len(image_files_2d)} static images: {output_video_2d}")
            else:
                print("No 2D static images found to create animation.")
        
        # Path to 3D static images
        static3D_dir = os.path.join(output_folder, "static3Dplots")
        if os.path.exists(static3D_dir):
            # Get all 3D surface PNG files and sort them by time step
            image_files_3d = sorted(
                [os.path.join(static3D_dir, f) for f in os.listdir(static3D_dir) if f.startswith('3d_surface_t') and f.endswith('.png')],
                key=lambda x: int(os.path.basename(x).split('t')[1].split('.')[0])  # Extract time step number for sorting
            )
            
            if image_files_3d:
                # Create output video file path
                output_video_3d = os.path.join(video_dir, "3d_surface_animation.mp4")
                
                # Load images and create video
                with imageio.get_writer(output_video_3d, fps=4) as writer:
                    for image_file in image_files_3d:
                        image = imageio.imread(image_file)
                        writer.append_data(image)
                
                print(f"Created 3D animation from {len(image_files_3d)} static images: {output_video_3d}")
            else:
                print("No 3D static images found to create animation.")
                
    except ImportError:
        print("Could not import imageio. Falling back to matplotlib animation...")
        # Fallback to the original animation creation using matplotlib if imageio is not available
        
        # Create 2D animation using matplotlib (fallback method)
        fig_2d, ax_2d = plt.subplots(figsize=(8, 6))
        
        def animate_2d(t):
            ax_2d.clear()
            FR = FR_time[t]
            im = ax_2d.imshow(FR, origin='lower', aspect='auto',
                             extent=[min(beta_values), max(beta_values), min(spectral_radius_values), max(spectral_radius_values)],
                             cmap='viridis')
            if t == 0:  # Only add colorbar on first frame
                plt.colorbar(im, label='Avg Firing Rate')
            ax_2d.set_xlabel('Beta Reservoir')
            ax_2d.set_ylabel('Spectral Radius')
            ax_2d.set_title(f"2D Heatmap at Time Step {t}")
            return [im]
        
        anim_2d = animation.FuncAnimation(fig_2d, animate_2d, frames=T, interval=250, blit=True)
        anim_2d.save(os.path.join(video_dir, "2d_heatmap_animation_fallback.mp4"), writer='ffmpeg', fps=4)
        plt.close(fig_2d)
        
        # Create 3D animation using matplotlib (fallback method)
        fig_3d = plt.figure(figsize=(10, 8))
        ax_3d = fig_3d.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(beta_values, spectral_radius_values)
        
        def animate_3d(t):
            ax_3d.clear()
            FR = FR_time[t]
            surf = ax_3d.plot_surface(X, Y, FR, cmap='viridis', edgecolor='none')
            ax_3d.view_init(elev=30, azim=120)
            ax_3d.set_xlabel('Beta Reservoir')
            ax_3d.set_ylabel('Spectral Radius')
            ax_3d.set_zlabel('Avg Firing Rate')
            ax_3d.set_title(f"3D Surface at Time Step {t}")
            return [surf]
        
        anim_3d = animation.FuncAnimation(fig_3d, animate_3d, frames=T, interval=250, blit=False)
        anim_3d.save(os.path.join(video_dir, "3d_surface_animation_fallback.mp4"), writer='ffmpeg', fps=4)
        plt.close(fig_3d)
        
    except Exception as e:
        print(f"Error creating animations from static images: {str(e)}")