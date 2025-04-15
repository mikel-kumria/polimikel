import os
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
from typing import Dict, List, Tuple, Any
import glob

def load_experiment_data(base_dir: str) -> Dict[str, Any]:
    """
    Load all experiment data from the base directory.
    Returns a dictionary with organized data and parameter combinations.
    """
    data = {
        'parameters': {
            'spectral_radius': set(),
            'topology': set(),
            'sparsity': set(),
            'matrix_seed': set(),
            'input_seed': set(),
            'reset_mechanism': set()
        },
        'results': {}
    }
    
    # Walk through all experiment directories
    for root, _, files in os.walk(base_dir):
        if 'FR_time.npy' in files and 'hyperparameters.json' in files:
            # Load hyperparameters
            with open(os.path.join(root, 'hyperparameters.json'), 'r') as f:
                hyperparams = json.load(f)
            
            # Load FR_time data
            fr_time = np.load(os.path.join(root, 'FR_time.npy'))
            
            # Extract parameters from hyperparameters
            params = {
                'spectral_radius': hyperparams.get('spectral_radius', None),
                'topology': hyperparams.get('topology', None),
                'sparsity': hyperparams.get('sparsity', None),
                'matrix_seed': hyperparams.get('matrix_seed', None),
                'input_seed': hyperparams.get('input_seed', None),
                'reset_mechanism': hyperparams.get('reset_mechanism', None),
                'threshold_range': hyperparams.get('threshold_range', [0.0, 2.0]),
                'beta_reservoir_range': hyperparams.get('beta_reservoir_range', [0.0, 1.0]),
                'grid_points': hyperparams.get('grid_points', 5)
            }
            
            # Update parameter sets
            for key in data['parameters'].keys():
                if params[key] is not None:
                    data['parameters'][key].add(params[key])
            
            # Create a unique key for this parameter combination
            param_key = (
                f"rho{params['spectral_radius']}_"
                f"{params['topology']}_"
                f"sparsity{params['sparsity']}_"
                f"mseed{params['matrix_seed']}_"
                f"iseed{params['input_seed']}_"
                f"reset{params['reset_mechanism']}"
            )
            
            # Store results
            data['results'][param_key] = {
                'fr_time': fr_time,
                'params': params
            }
    
    # Convert sets to sorted lists
    for key in data['parameters'].keys():
        data['parameters'][key] = sorted(list(data['parameters'][key]))
    
    return data

def compute_convergence_times(fr_time: np.ndarray) -> np.ndarray:
    """
    Compute convergence times for each parameter combination.
    """
    T, n_thresh, n_beta = fr_time.shape
    convergence_times = np.full((n_thresh, n_beta), T)
    
    for i in range(n_thresh):
        for j in range(n_beta):
            firing_rates = fr_time[:, i, j]
            below_threshold = np.where(firing_rates < 0.01)[0]
            if len(below_threshold) > 0:
                convergence_times[i, j] = below_threshold[0]
    
    return convergence_times

def create_parameter_buttons(data: Dict[str, Any], current_params: Dict[str, Any], x_pos: float = 1.2) -> List[Dict]:
    """
    Create dropdown buttons for each parameter in a vertical column.
    x_pos: position of buttons (1.2 means 20% to the right of the plot)
    """
    buttons = []
    button_names = [
        ('reset_mechanism', 'Reset Mechanism'),
        ('topology', 'Topology'),
        ('spectral_radius', 'Spectral Radius'),
        ('sparsity', 'Sparsity'),
        ('matrix_seed', 'Matrix Seed'),
        ('input_seed', 'Input Seed')
    ]
    
    for i, (param_name, display_name) in enumerate(button_names):
        if param_name in data['parameters']:
            dropdown = {
                'buttons': [
                    {
                        'args': [{'visible': [
                            all(
                                p == current_params[k] if k != param_name else p == value
                                for k, p in zip(current_params.keys(), key.split('_'))
                            )
                            for key in data['results'].keys()
                        ]}],
                        'label': str(value),
                        'method': 'update'
                    }
                    for value in sorted(data['parameters'][param_name])
                ],
                'direction': 'down',
                'showactive': True,
                'x': x_pos,  # Position to the right of the plot
                'xanchor': 'left',
                'y': 0.9 - (i * 0.15),  # Stack buttons vertically
                'yanchor': 'top',
                'name': display_name,
                'bgcolor': 'lightgray',
                'font': {'size': 12}
            }
            buttons.append(dropdown)
    
    return buttons

def create_convergence_plot(data: Dict[str, Any], first_result: Dict[str, Any], output_file: str):
    """Create the convergence time heatmap plot."""
    fr_time = first_result['fr_time']
    params = first_result['params']
    
    # Create parameter grids
    n_points = params['grid_points']
    threshold_vals = np.linspace(params['threshold_range'][0], params['threshold_range'][1], n_points)
    beta_vals = np.linspace(params['beta_reservoir_range'][0], params['beta_reservoir_range'][1], n_points)
    
    # Compute convergence times
    conv_times = compute_convergence_times(fr_time)
    
    # Create figure
    fig = go.Figure()
    
    # Add convergence time heatmap
    fig.add_trace(
        go.Heatmap(
            z=conv_times,
            x=beta_vals,
            y=threshold_vals,
            colorscale='Viridis',
            name='Convergence Time'
        )
    )
    
    # Update layout with parameter selection buttons
    current_params = {
        'spectral_radius': params['spectral_radius'],
        'topology': params['topology'],
        'sparsity': params['sparsity'],
        'matrix_seed': params['matrix_seed'],
        'input_seed': params['input_seed'],
        'reset_mechanism': params['reset_mechanism']
    }
    
    buttons = create_parameter_buttons(data, current_params)
    
    # Update layout
    fig.update_layout(
        title='Convergence Time Analysis',
        xaxis_title='Beta Reservoir',
        yaxis_title='V_threshold',
        updatemenus=buttons,
        height=800,
        width=1200,
        margin=dict(l=200)  # Add left margin for buttons
    )
    
    # Save the plot
    fig.write_html(output_file)

def create_2d_heatmap(data: Dict[str, Any], first_result: Dict[str, Any], output_file: str):
    """Create the 2D average firing rate heatmap plot with time evolution."""
    fr_time = first_result['fr_time']
    params = first_result['params']
    
    # Create parameter grids
    n_points = params['grid_points']
    threshold_vals = np.linspace(params['threshold_range'][0], params['threshold_range'][1], n_points)
    beta_vals = np.linspace(params['beta_reservoir_range'][0], params['beta_reservoir_range'][1], n_points)
    
    # Create figure
    fig = go.Figure()
    
    # Add frames for time evolution
    frames = []
    for t in range(fr_time.shape[0]):
        frame = go.Frame(
            data=[go.Heatmap(
                z=fr_time[t],
                x=beta_vals,
                y=threshold_vals,
                colorscale='Viridis',
                name=f'Time step {t}'
            )],
            name=str(t)
        )
        frames.append(frame)
    
    # Add initial heatmap
    fig.add_trace(
        go.Heatmap(
            z=fr_time[0],
            x=beta_vals,
            y=threshold_vals,
            colorscale='Viridis',
            name='Firing Rate'
        )
    )
    
    # Add frames to figure
    fig.frames = frames
    
    # Create slider
    sliders = [{
        'active': 0,
        'yanchor': 'top',
        'xanchor': 'left',
        'currentvalue': {
            'font': {'size': 16},
            'prefix': 'Time Step: ',
            'visible': True,
            'xanchor': 'right'
        },
        'transition': {'duration': 100},
        'pad': {'b': 10, 't': 50},
        'len': 0.9,
        'x': 0.1,
        'y': 0,
        'steps': [
            {
                'args': [
                    [str(t)],
                    {
                        'frame': {'duration': 100, 'redraw': True},
                        'mode': 'immediate',
                        'transition': {'duration': 100}
                    }
                ],
                'label': str(t),
                'method': 'animate'
            }
            for t in range(fr_time.shape[0])
        ]
    }]
    
    # Add play and pause buttons
    updatemenus = [
        {
            'type': 'buttons',
            'showactive': False,
            'x': 0.1,
            'y': 1.15,
            'xanchor': 'right',
            'yanchor': 'top',
            'buttons': [
                {
                    'label': 'Play',
                    'method': 'animate',
                    'args': [
                        None,
                        {
                            'frame': {'duration': 100, 'redraw': True},
                            'fromcurrent': True,
                            'transition': {'duration': 100}
                        }
                    ]
                },
                {
                    'label': 'Pause',
                    'method': 'animate',
                    'args': [
                        [None],
                        {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }
                    ]
                }
            ]
        }
    ]
    
    # Update layout with parameter selection buttons
    current_params = {
        'spectral_radius': params['spectral_radius'],
        'topology': params['topology'],
        'sparsity': params['sparsity'],
        'matrix_seed': params['matrix_seed'],
        'input_seed': params['input_seed'],
        'reset_mechanism': params['reset_mechanism']
    }
    
    # Add parameter buttons to updatemenus
    updatemenus.extend(create_parameter_buttons(data, current_params))
    
    # Update layout
    fig.update_layout(
        title='Firing Rate Evolution Over Time',
        xaxis_title='Beta Reservoir',
        yaxis_title='V_threshold',
        updatemenus=updatemenus,
        sliders=sliders,
        height=800,
        width=1200,
        margin=dict(r=250, t=100),  # Increase right margin for buttons
        showlegend=False
    )
    
    # Save the plot
    fig.write_html(output_file)

def create_3d_surface(data: Dict[str, Any], first_result: Dict[str, Any], output_file: str):
    """Create the 3D average firing rate surface plot."""
    fr_time = first_result['fr_time']
    params = first_result['params']
    
    # Create parameter grids
    n_points = params['grid_points']
    threshold_vals = np.linspace(params['threshold_range'][0], params['threshold_range'][1], n_points)
    beta_vals = np.linspace(params['beta_reservoir_range'][0], params['beta_reservoir_range'][1], n_points)
    
    # Calculate average firing rate
    avg_fr = np.mean(fr_time, axis=0)
    
    # Create figure
    fig = go.Figure()
    
    # Add 3D surface
    fig.add_trace(
        go.Surface(
            z=avg_fr,
            x=beta_vals,
            y=threshold_vals,
            colorscale='Viridis',
            name='3D Average Firing Rate'
        )
    )
    
    # Update layout with parameter selection buttons
    current_params = {
        'spectral_radius': params['spectral_radius'],
        'topology': params['topology'],
        'sparsity': params['sparsity'],
        'matrix_seed': params['matrix_seed'],
        'input_seed': params['input_seed'],
        'reset_mechanism': params['reset_mechanism']
    }
    
    buttons = create_parameter_buttons(data, current_params)
    
    # Update layout
    fig.update_layout(
        title='Average Firing Rate (3D Surface)',
        scene=dict(
            xaxis_title='Beta Reservoir',
            yaxis_title='V_threshold',
            zaxis_title='Average Firing Rate'
        ),
        updatemenus=buttons,
        height=800,
        width=1200,
        margin=dict(l=200)  # Add left margin for buttons
    )
    
    # Save the plot
    fig.write_html(output_file)

def create_interactive_plots(base_dir: str):
    """
    Create three separate interactive plots with parameter selection buttons.
    """
    # Load all experiment data
    data = load_experiment_data(base_dir)
    
    if not data['results']:
        print("No experiment data found!")
        return
    
    # Get first result for initialization
    first_key = list(data['results'].keys())[0]
    first_result = data['results'][first_key]
    
    # Create output files
    convergence_file = os.path.join(base_dir, "convergence_time.html")
    heatmap_file = os.path.join(base_dir, "firing_rate_2d.html")
    surface_file = os.path.join(base_dir, "firing_rate_3d.html")
    
    # Create each plot
    create_convergence_plot(data, first_result, convergence_file)
    create_2d_heatmap(data, first_result, heatmap_file)
    create_3d_surface(data, first_result, surface_file)
    
    print(f"Created interactive plots:")
    print(f"  Convergence Time: {convergence_file}")
    print(f"  2D Firing Rate: {heatmap_file}")
    print(f"  3D Firing Rate: {surface_file}")

def main():
    """
    Main function to create interactive plots from parallel experiment results.
    """
    # Get the latest experiment directory
    base_dir = "/Users/mikel/Documents/GitHub/polimikel/UCR/Simulations"
    experiment_dirs = glob.glob(os.path.join(base_dir, "parallel_experiment_*"))
    if not experiment_dirs:
        print("No experiment directories found!")
        return
    
    latest_dir = max(experiment_dirs, key=os.path.getctime)
    print(f"Creating interactive plots from: {latest_dir}")
    create_interactive_plots(latest_dir)

if __name__ == "__main__":
    main() 