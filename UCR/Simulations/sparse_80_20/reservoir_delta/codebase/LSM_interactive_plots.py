import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

# Dummy experiment configuration for testing
EXPERIMENT_CONFIG = {
    "parameters": {
        "reset_mechanisms": ["zero", "subtract"],
        "topologies": {
            "Random": ["80_20"],
            "Dale": ["80_20"]
        },
        "spectral_radii": ["rho0x5", "rho1x0"],
        "sparsities": ["0.1", "0.9"],
        "matrix_seeds": [1, 2],
        "input_seeds": [1, 2]
    },
    "simulation": {
        "threshold_range": [0.0, 2.0],
        "beta_reservoir_range": [0.0, 1.0],
        "grid_points": 5
    }
}

class InteractivePlotter:
    def __init__(self, fr_time_paths):
        """
        Initialize the interactive plotter with paths to FR_time.npy files.
        
        Args:
            fr_time_paths (list): List of paths to FR_time.npy files
        """
        self.fr_time_data = {}
        self.beta_values = np.linspace(
            EXPERIMENT_CONFIG["simulation"]["beta_reservoir_range"][0],
            EXPERIMENT_CONFIG["simulation"]["beta_reservoir_range"][1],
            EXPERIMENT_CONFIG["simulation"]["grid_points"]
        )
        self.threshold_values = np.linspace(
            EXPERIMENT_CONFIG["simulation"]["threshold_range"][0],
            EXPERIMENT_CONFIG["simulation"]["threshold_range"][1],
            EXPERIMENT_CONFIG["simulation"]["grid_points"]
        )
        
        # Load FR_time data from each path
        for path in fr_time_paths:
            # Extract trial number from path
            trial_num = os.path.basename(os.path.dirname(path)).split('_')[1]
            self.fr_time_data[f"Trial {trial_num}"] = np.load(path)
    
    def compute_convergence_times(self, fr_time):
        """Compute convergence times for each parameter combination."""
        T, n_thresh, n_beta = fr_time.shape
        convergence_times = np.full((n_thresh, n_beta), T)
        
        for i in range(n_thresh):
            for j in range(n_beta):
                firing_rates = fr_time[:, i, j]
                below_threshold = np.where(firing_rates < 0.01)[0]
                if len(below_threshold) > 0:
                    convergence_times[i, j] = below_threshold[0]
        
        return convergence_times
    
    def create_right_side_dropdowns(self):
        """Create standardized right-side dropdown menus."""
        # Create dropdown options
        dropdown_options = [
            ("Trial", list(self.fr_time_data.keys())),
            ("Reset Mechanism", ["zero", "subtract", "none"]),
            ("Topology", ["Random", "Dale", "Small_world"]),
            ("Spectral Radius", ["0.1", "0.5", "1.0", "1.5", "2.0"]),
            ("Sparsity", ["0.1", "0.5", "0.9", "1.0"]),
            ("Matrix Seed", ["1", "2", "3", "4", "5"]),
            ("Input Seed", ["1", "2", "3", "4", "5"])
        ]
        
        updatemenus = []
        annotations = []
        
        # Create dropdown menus
        for i, (param_name, options) in enumerate(dropdown_options):
            # For the first dropdown (Trial selection), create working buttons
            if i == 0:
                buttons = [dict(
                    args=[{"visible": [j == k for j in range(len(options))]}],
                    label=option,
                    method="update"
                ) for k, option in enumerate(options)]
            else:
                # For other dropdowns, create dummy buttons
                buttons = [dict(
                    args=[{}],
                    label=option,
                    method="skip"
                ) for option in options]
            
            # Add the dropdown menu
            updatemenus.append(dict(
                buttons=buttons,
                direction="down",
                showactive=True,
                x=1.25,  # Position further to the right to avoid overlap
                y=0.95 - i * 0.13,  # Increase spacing between dropdowns
                xanchor="left",
                yanchor="top",
                bgcolor='lightgray',
                bordercolor='#AAAAAA',
                font=dict(size=12),
                pad=dict(t=5, b=5, r=5, l=5)
            ))
            
            # Add a label for the dropdown
            annotations.append(dict(
                x=1.24,  # Position the label slightly to the left of the dropdowns
                y=0.95 - i * 0.13,
                xref="paper",
                yref="paper",
                text=param_name,
                showarrow=False,
                font=dict(size=12, color='#444444')
            ))
        
        return updatemenus, annotations
    
    def create_convergence_plot(self, output_file='convergence_heatmap.html'):
        """Create the convergence time heatmap plot with right-side dropdowns."""
        fig = go.Figure()
        
        # Add traces for each trial
        for trial_name, fr_time in self.fr_time_data.items():
            conv_times = self.compute_convergence_times(fr_time)
            
            fig.add_trace(go.Heatmap(
                z=conv_times,
                x=self.beta_values,
                y=self.threshold_values,
                colorscale='Viridis',
                name=trial_name,
                visible=False,
                colorbar=dict(
                    title='Time Steps',
                    thickness=15,
                    len=0.7,
                    x=1.02,  # Move colorbar to the left to avoid overlap with dropdowns
                    title_side='right'
                )
            ))
        
        # Make first trace visible
        fig.data[0].visible = True
        
        # Create standardized right-side dropdowns
        updatemenus, annotations = self.create_right_side_dropdowns()
        
        # Update layout
        fig.update_layout(
            title='Convergence Time Heatmap',
            title_font=dict(size=18),
            xaxis_title='Beta Reservoir',
            yaxis_title='V_threshold',
            height=800,
            width=1200,
            margin=dict(r=300, t=50, l=80, b=80),  # Increase right margin significantly
            updatemenus=updatemenus,
            annotations=annotations,
            paper_bgcolor='rgba(245,245,245,1)',
            plot_bgcolor='rgba(245,245,245,1)',
            font=dict(family="Arial, sans-serif", size=14),
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Arial, sans-serif"
            )
        )
        
        fig.write_html(output_file)
    
    def create_2d_firing_rate_plot(self, output_file='firing_rate_2d.html'):
        """Create the 2D firing rate heatmap plot with animation and right-side dropdowns."""
        fig = go.Figure()
        
        # Add initial heatmap for each trial
        for trial_name, fr_time in self.fr_time_data.items():
            fig.add_trace(go.Heatmap(
                z=fr_time[0],
                x=self.beta_values,
                y=self.threshold_values,
                colorscale='Viridis',
                name=trial_name,
                visible=False,
                colorbar=dict(
                    title='Firing Rate',
                    thickness=15,
                    len=0.7,
                    x=1.02,
                    title_side='right'
                ),
                # Set showscale to False for all but the first one
                showscale=False
            ))
        
        # Make first trace visible and show its colorbar
        fig.data[0].visible = True
        fig.data[0].showscale = True
        
        # Create frames for animation without colorbars in the frame data
        frames = []
        for t in range(self.fr_time_data[list(self.fr_time_data.keys())[0]].shape[0]):
            frame_data = []
            for i, fr_time in enumerate(self.fr_time_data.values()):
                frame_data.append(go.Heatmap(
                    z=fr_time[t],
                    x=self.beta_values,
                    y=self.threshold_values,
                    colorscale='Viridis',
                    showscale=i == 0,  # Only show scale for the first trace
                    colorbar=dict(
                        title='Firing Rate',
                        thickness=15,
                        len=0.7,
                        x=1.02,
                        title_side='right'
                    )
                ))
            frames.append(go.Frame(data=frame_data, name=str(t)))
        
        fig.frames = frames
        
        # Create standardized right-side dropdowns
        updatemenus, annotations = self.create_right_side_dropdowns()
        
        # Add animation controls to updatemenus
        animation_buttons = dict(
            type="buttons",
            showactive=False,
            x=0.15,
            y=1.05,
            xanchor="left",
            yanchor="top",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, {
                        "frame": {"duration": 100, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 100}
                    }]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }]
                )
            ],
            bgcolor='rgba(240,240,240,0.8)',
            bordercolor='#999999',
            font=dict(size=13)
        )
        updatemenus.append(animation_buttons)
        
        # Create slider
        sliders = [{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 14, 'color': '#444444'},
                'prefix': 'Time Step: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 100},
            'pad': {'b': 10, 't': 20, 'l': 20, 'r': 20},
            'len': 0.7,  # Make slider shorter to avoid overlap
            'x': 0.1,
            'y': 0,
            'steps': [
                {
                    'args': [
                        [str(t)],
                        {
                            'frame': {'duration': 100, "redraw": True},
                            'mode': 'immediate',
                            'transition': {'duration': 100}
                        }
                    ],
                    'label': str(t),
                    'method': 'animate'
                }
                for t in range(self.fr_time_data[list(self.fr_time_data.keys())[0]].shape[0])
            ],
            'bgcolor': 'rgba(240,240,240,0.8)',
            'bordercolor': '#999999',
            'ticklen': 5,
            'tickwidth': 2
        }]
        
        # Update layout with uirevision to maintain consistent colorbar across frames
        fig.update_layout(
            title='2D Firing Rate Heatmap',
            title_font=dict(size=18),
            xaxis_title='Beta Reservoir',
            yaxis_title='V_threshold',
            height=800,
            width=1200,
            margin=dict(r=300, t=80, l=80, b=120),
            updatemenus=updatemenus,
            annotations=annotations,
            sliders=sliders,
            paper_bgcolor='rgba(245,245,245,1)',
            plot_bgcolor='rgba(245,245,245,1)',
            font=dict(family="Arial, sans-serif", size=14),
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Arial, sans-serif"
            ),
            uirevision=True  # Important for maintaining colorbar consistency
        )
        
        fig.write_html(output_file)
    
    def create_3d_firing_rate_plot(self, output_file='firing_rate_3d.html'):
        """Create the 3D firing rate surface plot with animation and right-side dropdowns."""
        fig = go.Figure()
        
        # Add initial surface for each trial
        for trial_name, fr_time in self.fr_time_data.items():
            fig.add_trace(go.Surface(
                z=fr_time[0],
                x=self.beta_values,
                y=self.threshold_values,
                colorscale='Viridis',
                name=trial_name,
                visible=False,
                colorbar=dict(
                    title='Firing Rate',
                    thickness=15,
                    len=0.7,
                    x=1.02,
                    title_side='right'
                ),
                lighting=dict(
                    ambient=0.6,
                    diffuse=0.8,
                    roughness=0.5,
                    specular=0.4
                ),
                contours=dict(
                    z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
                ),
                # Set showscale to False for all but the first one
                showscale=False
            ))
        
        # Make first trace visible and show its colorbar
        fig.data[0].visible = True
        fig.data[0].showscale = True
        
        # Create frames for animation without colorbars in the frame data
        frames = []
        for t in range(self.fr_time_data[list(self.fr_time_data.keys())[0]].shape[0]):
            frame_data = []
            for i, fr_time in enumerate(self.fr_time_data.values()):
                frame_data.append(go.Surface(
                    z=fr_time[t],
                    x=self.beta_values,
                    y=self.threshold_values,
                    colorscale='Viridis',
                    showscale=i == 0,  # Only show scale for the first trace
                    colorbar=dict(
                        title='Firing Rate',
                        thickness=15,
                        len=0.7,
                        x=1.02,
                        title_side='right'
                    ),
                    lighting=dict(
                        ambient=0.6,
                        diffuse=0.8,
                        roughness=0.5,
                        specular=0.4
                    ),
                    contours=dict(
                        z=dict(show=True, usecolormap=True, highlightcolor="white", project_z=True)
                    )
                ))
            frames.append(go.Frame(data=frame_data, name=str(t)))
        
        fig.frames = frames
        
        # Create standardized right-side dropdowns
        updatemenus, annotations = self.create_right_side_dropdowns()
        
        # Add animation controls to updatemenus
        animation_buttons = dict(
            type="buttons",
            showactive=False,
            x=0.15,
            y=1.05,
            xanchor="left",
            yanchor="top",
            buttons=[
                dict(
                    label="Play",
                    method="animate",
                    args=[None, {
                        "frame": {"duration": 100, "redraw": True},
                        "fromcurrent": True,
                        "transition": {"duration": 100}
                    }]
                ),
                dict(
                    label="Pause",
                    method="animate",
                    args=[[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate",
                        "transition": {"duration": 0}
                    }]
                )
            ],
            bgcolor='rgba(240,240,240,0.8)',
            bordercolor='#999999',
            font=dict(size=13)
        )
        updatemenus.append(animation_buttons)
        
        # Create slider
        sliders = [{
            'active': 0,
            'yanchor': 'top',
            'xanchor': 'left',
            'currentvalue': {
                'font': {'size': 14, 'color': '#444444'},
                'prefix': 'Time Step: ',
                'visible': True,
                'xanchor': 'right'
            },
            'transition': {'duration': 100},
            'pad': {'b': 10, 't': 20, 'l': 20, 'r': 20},
            'len': 0.7,  # Make slider shorter to avoid overlap
            'x': 0.1,
            'y': 0,
            'steps': [
                {
                    'args': [
                        [str(t)],
                        {
                            'frame': {'duration': 100, "redraw": True},
                            'mode': 'immediate',
                            'transition': {'duration': 100}
                        }
                    ],
                    'label': str(t),
                    'method': 'animate'
                }
                for t in range(self.fr_time_data[list(self.fr_time_data.keys())[0]].shape[0])
            ],
            'bgcolor': 'rgba(240,240,240,0.8)',
            'bordercolor': '#999999',
            'ticklen': 5,
            'tickwidth': 2
        }]
        
        # Update layout with uirevision to maintain consistent colorbar across frames
        fig.update_layout(
            title='3D Firing Rate Surface',
            title_font=dict(size=18),
            scene=dict(
                xaxis_title='Beta Reservoir',
                yaxis_title='V_threshold',
                zaxis_title='Firing Rate',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                ),
                aspectratio=dict(x=1, y=1, z=0.7)
            ),
            height=800,
            width=1200,
            margin=dict(r=300, t=80, l=10, b=10, pad=0),
            updatemenus=updatemenus,
            annotations=annotations,
            sliders=sliders,
            paper_bgcolor='rgba(245,245,245,1)',
            font=dict(family="Arial, sans-serif", size=14),
            hoverlabel=dict(
                bgcolor="white",
                font_size=14,
                font_family="Arial, sans-serif"
            ),
            uirevision=True  # Important for maintaining colorbar consistency
        )
        
        fig.write_html(output_file)

def main():
    # Test FR_time paths
    fr_time_paths = [
        "/Users/mikel/Documents/GitHub/polimikel/UCR/Simulations/sparse_80_20/reservoir_delta/results/old_results/trial_10_date_2025_04_14_15_09/FR_time.npy",
        "/Users/mikel/Documents/GitHub/polimikel/UCR/Simulations/sparse_80_20/reservoir_delta/results/old_results/trial_9_date_2025_04_14_14_29/FR_time.npy"
    ]
    
    # Create plotter and generate interactive visualizations
    plotter = InteractivePlotter(fr_time_paths)
    
    # Create each plot type
    plotter.create_convergence_plot()
    plotter.create_2d_firing_rate_plot()
    plotter.create_3d_firing_rate_plot()
    
    print("Interactive plots have been created:")
    print("1. convergence_heatmap.html - Convergence time heatmap")
    print("2. firing_rate_2d.html - 2D firing rate heatmap with animation")
    print("3. firing_rate_3d.html - 3D firing rate surface with animation")

if __name__ == "__main__":
    main() 