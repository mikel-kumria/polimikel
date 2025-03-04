import torch
import matplotlib.pyplot as plt
import numpy as np

def compute_sharpness(model, inputs, targets, loss_function, epsilon=1e-3):
    # Assume a single batch of inputs/targets for simplicity
    original_weights = [p.clone().detach() for p in model.parameters()]

    # Perturb weights
    for p in model.parameters():
        if p.requires_grad:
            perturbation = torch.randn_like(p) * epsilon
            p.data.add_(perturbation)

    with torch.no_grad():
        # Compute perturbed loss
        spk1, spk2, spk3, spk4, mem4 = model(inputs)
        final_out = mem4.sum(0)
        perturbed_loss = loss_function(final_out, targets).item()

    # Restore original weights
    for p, w in zip(model.parameters(), original_weights):
        p.data.copy_(w)

    # Compute original loss
    with torch.no_grad():
        spk1, spk2, spk3, spk4, mem4 = model(inputs)
        final_out = mem4.sum(0)  # same logic as training and perturbed
        original_loss = loss_function(final_out, targets).item()

    return perturbed_loss - original_loss

def compute_gradient_noise(model, dataloader, loss_function, device='cpu', num_batches=10):
    # Compute variance of gradients over a few mini-batches
    grad_list = []
    model.eval()
    count = 0
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        model.zero_grad()
        spk1, spk2, spk3, spk4, mem4 = model(inputs)
        final_out = mem4.sum(0)  # same logic as training
        loss = loss_function(final_out, targets)
        loss.backward()

        grad_vector = torch.cat([p.grad.view(-1) for p in model.parameters() if p.grad is not None])
        grad_list.append(grad_vector.detach().cpu())
        count += 1
        if count >= num_batches:
            break
    if len(grad_list) < 2:
        return 0.0
    grad_matrix = torch.stack(grad_list)
    mean_grad = grad_matrix.mean(dim=0)
    grad_noise = ((grad_matrix - mean_grad)**2).mean().item()
    return grad_noise

def visualize_loss_landscape(model, inputs, targets, loss_function, d1_scale=0.1, d2_scale=0.1, steps=5):
    # Simple visualization of a 2D slice of loss landscape
    model.eval()
    directions = []
    for p in model.parameters():
        if p.requires_grad:
            directions.append(torch.randn_like(p))
    d1 = [d * d1_scale for d in directions]
    d2 = [d * d2_scale for d in directions]

    alphas = np.linspace(-1, 1, steps)
    betas = np.linspace(-1, 1, steps)
    loss_surface = np.zeros((steps, steps))

    original_weights = [p.clone().detach() for p in model.parameters()]

    with torch.no_grad():
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                for p, w, dw1, dw2 in zip(model.parameters(), original_weights, d1, d2):
                    p.data = w + alpha * dw1 + beta * dw2
                outputs = model(inputs)
                loss_surface[i, j] = loss_function(outputs, targets).item()

    # Restore original weights
    for p, w in zip(model.parameters(), original_weights):
        p.data.copy_(w)

    fig, ax = plt.subplots()
    cs = ax.contourf(alphas, betas, loss_surface, levels=20, cmap='viridis')
    fig.colorbar(cs)
    ax.set_title('Loss Landscape')
    ax.set_xlabel('Direction 1')
    ax.set_ylabel('Direction 2')
    return fig