import os
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
import ray
from ray import tune, train
from ray.tune.search.optuna import OptunaSearch
from ray.tune.schedulers import MedianStoppingRule
from functools import partial

from exp_libs import (
    CochleaDataset,
    Lightning_SNNQUT_no_cochlea,
    generate_tau_beta_values,
    finalize_quantization,
)


def train_net(config, data_dir, num_epochs=10, concurrent_trials=8):

    process_memory = 0.90 / concurrent_trials
    # Configure PyTorch to use a fraction of GPU memory
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(process_memory)

    # Constants (same as original)
    input_size = 16
    hidden_size = 24
    output_size = 4
    batch_size = 32
    scheduler_step_size = 1000000000
    scheduler_gamma = 0.5
    num_workers_loader = max(1, os.cpu_count() - 1)
    hidden_reset_mechanism = "zero"
    output_reset_mechanism = "zero"
    delta_t = 1

    # Load cochlea weights (now using absolute path) ONLY FOR COCHLEA VERSION
    # loaded_weights = np.load(os.path.join(weights_dir, "weights.npy"))
    # loaded_weights = torch.from_numpy(loaded_weights)

    # Generate beta values
    beta_hidden_1, beta_hidden_2, beta_hidden_3, beta_output = generate_tau_beta_values(
        hidden_size, output_size, delta_t=delta_t
    )

    # Set random seed
    pl.seed_everything(42)

    # Initialize datasets and dataloaders
    train_dataset = CochleaDataset(data_dir, split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers_loader,
        pin_memory=True,
        persistent_workers=True,
    )

    # Initialize model
    model = Lightning_SNNQUT_no_cochlea(
        input_size=input_size,
        hidden_size=hidden_size,
        output_size=output_size,
        beta_hidden_1=beta_hidden_1,
        beta_hidden_2=beta_hidden_2,
        beta_hidden_3=beta_hidden_3,
        beta_output=beta_output,
        hidden_reset_mechanism=hidden_reset_mechanism,
        output_reset_mechanism=output_reset_mechanism,
        learning_rate=config["learning_rate"],
        optimizer_betas=config["optimizer_betas"],
        scheduler_step_size=scheduler_step_size,
        scheduler_gamma=scheduler_gamma,
        output_threshold=config["hidden_threshold"],
        hidden_threshold=config["hidden_threshold"],
        fast_sigmoid_slope=config["fast_sigmoid_slope"],
        bits=config["bits"],
    )

    # Assign the pretrained cochlea weights
    # model.model.ch_fc.weight.data = loaded_weights.to(model.device)

    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        log_every_n_steps=1,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_progress_bar=True,
    )

    # Train the model
    trainer.fit(model, train_loader)

    train.report(
        {
            "loss": trainer.callback_metrics["train_loss"].item(),
            "accuracy": trainer.callback_metrics["train_accuracy"].item(),
        }
    )


def main():

    concurrent_trials = 6

    # Get absolute paths, this is because ray runs in a different folder than script's
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(
        current_dir,
        "../data/mini_DATASET_FILTERBANK/OLD_NAME_FORMAT",
    )

    experiment_results = os.path.join(current_dir, "ray_results")

    # Define search space
    search_space = {
        #"learning_rate": tune.loguniform(1e-5, 1e-2),
        "learning_rate": 1e-3,
        "hidden_threshold": tune.uniform(0.5, 8),
        "fast_sigmoid_slope": 20,
        "optimizer_betas": (0.9, 0.99),
        "bits": 32,
    }

    # Configure the scheduler
    ray_scheduler = MedianStoppingRule(
        time_attr="training_iteration",
        metric="accuracy",
        mode="max",
        grace_period=3,
        min_samples_required=3,
        hard_stop=True,
    )

    # Configure the Optuna search algorithm
    optuna_search = OptunaSearch(metric="accuracy", mode="max")

    # Initialize Ray with dashboard
    context = ray.init()
    print(context.dashboard_url)

    # Run the hyperparameter optimization
    analysis = tune.run(
        partial(
            train_net,
            data_dir=data_dir,
            num_epochs=50,
            concurrent_trials=concurrent_trials,
        ),
        config=search_space,
        num_samples=50,
        scheduler=ray_scheduler,
        search_alg=optuna_search,
        resources_per_trial={
            "cpu": 1,
            "gpu": 1 / concurrent_trials if torch.cuda.is_available() else 0,
        },
        name="modMARCO_LIF_trial_2",
        storage_path=experiment_results,
        verbose=1,
        resume=True,
        reuse_actors=False,
    )

    # Print the best hyperparameters
    best_trial = analysis.get_best_trial("accuracy", "max")
    print("Best trial config:", best_trial.config)
    print("Best trial final accuracy:", best_trial.last_result["accuracy"])


if __name__ == "__main__":
    main()
