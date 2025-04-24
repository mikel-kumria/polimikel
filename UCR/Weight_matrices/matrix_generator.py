import os
import json
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def create_directory(dir_path):
    """Create directory if it does not exist."""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def scale_to_radius(W, desired_radius):
    """
    Rescale the matrix W to have the desired spectral radius.
    
    Parameters:
        W (np.ndarray): The weight matrix.
        desired_radius (float): Target spectral radius.
        
    Returns:
        W_rescaled (np.ndarray): The rescaled matrix,
        current_radius (float): The original spectral radius,
        scaling_factor (float): The factor applied.
    """
    eigenvalues = np.linalg.eigvals(W)
    current_radius = np.max(np.abs(eigenvalues))
    scaling_factor = desired_radius / current_radius if current_radius != 0 else 1.0
    W_rescaled = W * scaling_factor
    return W_rescaled, current_radius, scaling_factor

def generate_ginibre_matrix(n, sigma=1.0, seed=42):
    """Generate a random Ginibre matrix."""
    np.random.seed(seed)
    return sigma * np.random.randn(n, n)

def generate_symmetric_matrix(n, sigma=1.0, seed=42):
    """Generate a random symmetric matrix with real eigenvalues."""
    np.random.seed(seed)
    A = sigma * np.random.randn(n, n)
    return 0.5 * (A + A.T)  # Symmetrize

def plot_eigenvalues(W, title, save_path):
    """Plot eigenvalues in the complex plane."""
    eigenvalues = np.linalg.eigvals(W)
    plt.figure(figsize=(8, 8))
    plt.scatter(eigenvalues.real, eigenvalues.imag, color='blue', alpha=0.7)
    plt.title(title)
    plt.xlabel("Real Part")
    plt.ylabel("Imaginary Part")
    plt.grid(True)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    
    # Add a circle showing the spectral radius
    radius = np.max(np.abs(eigenvalues))
    circle = plt.Circle((0, 0), radius, fill=False, color='red', linestyle='--', linewidth=1)
    plt.gca().add_patch(circle)
    
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_histogram(W, title, save_path, bins=50):
    """Plot histogram of matrix values."""
    plt.figure(figsize=(8, 6))
    plt.hist(W.flatten(), bins=bins, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_statistics(matrix_type, matrices, save_path):
    """Plot statistics (mean and std) across multiple matrices."""
    # Convert list of matrices to numpy array for easier computation
    matrices_array = np.array(matrices)
    
    # Calculate mean and std across all matrices
    mean_matrix = np.mean(matrices_array, axis=0)
    std_matrix = np.std(matrices_array, axis=0)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot mean matrix
    im1 = ax1.imshow(mean_matrix, cmap='viridis')
    ax1.set_title(f'Mean Matrix ({matrix_type})')
    fig.colorbar(im1, ax=ax1)
    
    # Plot standard deviation matrix
    im2 = ax2.imshow(std_matrix, cmap='viridis')
    ax2.set_title(f'Standard Deviation Matrix ({matrix_type})')
    fig.colorbar(im2, ax=ax2)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def generate_matrices(matrix_type, n=100, seeds=range(1, 101), rho=1.0, output_dir="weight_matrices"):
    """
    Generate matrices for different seeds with fixed spectral radius.
    
    Parameters:
        matrix_type (str): Either "Ginibre" or "Symmetric"
        n (int): Matrix dimension
        seeds (list): List of seeds to use
        rho (float): Spectral radius to use
        output_dir (str): Base output directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_folder = os.path.join(output_dir, f"{matrix_type}_{timestamp}")
    create_directory(base_folder)
    
    # Create a global metadata file
    global_metadata = {
        "matrix_type": matrix_type,
        "dimension": n,
        "timestamp": timestamp,
        "seeds": list(seeds),
        "spectral_radius": rho
    }
    
    with open(os.path.join(base_folder, "metadata.json"), "w") as f:
        json.dump(global_metadata, f, indent=4)
    
    rho_str = f"rho{str(rho).replace('.', 'x')}"
    rho_folder = os.path.join(base_folder, rho_str)
    create_directory(rho_folder)
    
    # Create plots directory
    plots_folder = os.path.join(rho_folder, "plots")
    create_directory(plots_folder)
    
    # Metadata for this rho
    rho_metadata = {
        "spectral_radius": rho,
        "matrix_files": [],
        "eigenvalue_plots": [],
        "histogram_plots": []
    }
    
    # Store all matrices for statistics
    all_matrices = []
    
    # Generate matrices for each seed
    for seed in seeds:
        if matrix_type == "Ginibre":
            W_initial = generate_ginibre_matrix(n, sigma=1.0, seed=seed)
        elif matrix_type == "Symmetric":
            W_initial = generate_symmetric_matrix(n, sigma=1.0, seed=seed)
        else:
            raise ValueError(f"Unknown matrix type: {matrix_type}")
        
        # Rescale to desired spectral radius
        W_rescaled, original_radius, scaling_factor = scale_to_radius(W_initial, rho)
        all_matrices.append(W_rescaled)
        
        # Save the matrix
        matrix_filename = f"W_res_seed{seed}.npy"
        matrix_path = os.path.join(rho_folder, matrix_filename)
        np.save(matrix_path, W_rescaled)
        
        # Create eigenvalue plot
        eig_plot_path = os.path.join(plots_folder, f"eigenvalues_seed{seed}.png")
        plot_eigenvalues(
            W_rescaled, 
            f"{matrix_type} Matrix with Spectral Radius {rho} (Seed {seed})", 
            eig_plot_path
        )
        
        # Create histogram plot
        hist_plot_path = os.path.join(plots_folder, f"histogram_seed{seed}.png")
        plot_histogram(
            W_rescaled, 
            f"Weight Distribution for {matrix_type} Matrix (ρ={rho}, Seed {seed})", 
            hist_plot_path
        )
        
        # Add to metadata
        rho_metadata["matrix_files"].append({
            "seed": seed,
            "filename": matrix_filename,
            "original_spectral_radius": float(original_radius),
            "scaling_factor": float(scaling_factor),
            "final_spectral_radius": float(rho)
        })
        
        rho_metadata["eigenvalue_plots"].append(f"plots/eigenvalues_seed{seed}.png")
        rho_metadata["histogram_plots"].append(f"plots/histogram_seed{seed}.png")
        
        print(f"Generated {matrix_type} matrix with ρ={rho} for seed {seed}")
    
    # Create statistical plots
    stats_plot_path = os.path.join(plots_folder, "statistics.png")
    plot_statistics(matrix_type, all_matrices, stats_plot_path)
    
    # Save metadata for this rho
    with open(os.path.join(rho_folder, "info.json"), "w") as f:
        json.dump(rho_metadata, f, indent=4)
    
    print(f"All {matrix_type} matrices generated and saved in {base_folder}")
    return base_folder

if __name__ == "__main__":
    # Configuration
    n = 100  # Matrix dimension
    seeds = range(1, 101)  # Seeds from 1 to 100
    rho = 1.0  # Fixed spectral radius
    
    output_dir = "generated_matrices"
    
    # Create output directory
    create_directory(output_dir)
    
    # Generate Ginibre matrices
    ginibre_folder = generate_matrices("Ginibre", n, seeds, rho=rho, output_dir=output_dir)
    print(f"Ginibre matrices saved in: {ginibre_folder}")
    
    # Generate Symmetric matrices
    symmetric_folder = generate_matrices("Symmetric", n, seeds, rho=rho, output_dir=output_dir)
    print(f"Symmetric matrices saved in: {symmetric_folder}") 