#!/usr/bin/env python3
"""
Training script for Optimized DragonNet.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .model import OptimizedDragonNet, hsic_loss, neyman_loss, uncertainty_based_weights
from .noise_model import TruncatedDPGMM

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DragonNetTrainer:
    """Trainer for Optimized DragonNet."""

    def __init__(
        self,
        model: OptimizedDragonNet,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        hsic_weight: float = 0.1,
        neyman_weight: float = 1.0,
        uncertainty_beta: float = 1.0,
        use_uncertainty_weighting: bool = True,
        pyro_update_freq: int = 10,
    ):
        self.model = model.to(device)
        self.device = device
        self.hsic_weight = hsic_weight
        self.neyman_weight = neyman_weight
        self.uncertainty_beta = uncertainty_beta
        self.use_uncertainty_weighting = use_uncertainty_weighting
        self.pyro_update_freq = pyro_update_freq

        # Optimizer
        self.optimizer = optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCELoss()

        # Training state
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []

    def train_epoch(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_mse = 0.0
        total_prop = 0.0
        total_hsic = 0.0
        total_neyman = 0.0
        num_batches = 0

        for batch_idx, (x, t, y) in enumerate(train_loader):
            x, t, y = x.to(self.device), t.to(self.device), y.to(self.device)

            # Forward pass with uncertainty estimation for weighting
            outputs = self.model(
                x, return_uncertainty=self.use_uncertainty_weighting, mc_samples=10
            )

            # Compute losses
            # 1. Propensity loss
            g = outputs["g"]
            prop_loss = self.bce_loss(g, t)

            # 2. Baseline outcome loss
            m = outputs["m"]
            mse_loss = self.mse_loss(m, y)

            # 3. HSIC loss between representation splits
            phi_a, phi_b = outputs["phi_a"], outputs["phi_b"]
            hsic_val = hsic_loss(phi_a, phi_b)

            # 4. Neyman orthogonal loss for treatment effect
            tau = outputs["tau"]
            neyman_val = neyman_loss(y, t, m, g, tau)

            # 5. Uncertainty-based weighting if enabled
            if self.use_uncertainty_weighting and "tau_var" in outputs:
                tau_var = outputs["tau_var"]
                sample_weights = uncertainty_based_weights(
                    tau_var, beta=self.uncertainty_beta
                )
                # Apply weights to MSE and propensity losses
                weighted_mse = (sample_weights * (m - y) ** 2).mean()
                weighted_prop = (
                    -(t * torch.log(g + 1e-8) + (1 - t) * torch.log(1 - g + 1e-8))
                    * sample_weights
                ).mean()
                mse_loss = weighted_mse
                prop_loss = weighted_prop

            # Total loss
            loss = (
                mse_loss
                + prop_loss
                + self.hsic_weight * hsic_val
                + self.neyman_weight * neyman_val
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Update Pyro noise model if enabled
            if self.model.use_pyro and batch_idx % self.pyro_update_freq == 0:
                with torch.no_grad():
                    residuals = (y - m).detach().cpu().numpy()
                    if len(residuals) > 0:
                        self.model.noise_model.partial_fit(
                            residuals, num_steps=5, verbose=False
                        )

            # Accumulate statistics
            total_loss += loss.item()
            total_mse += mse_loss.item()
            total_prop += prop_loss.item()
            total_hsic += hsic_val.item()
            total_neyman += neyman_val.item()
            num_batches += 1

            if verbose and batch_idx % max(1, len(train_loader) // 10) == 0:
                logger.info(
                    f"  Batch {batch_idx}/{len(train_loader)}: "
                    f"Loss={loss.item():.4f}, MSE={mse_loss.item():.4f}, "
                    f"Prop={prop_loss.item():.4f}, HSIC={hsic_val.item():.4f}"
                )

        # Compute epoch averages
        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_prop = total_prop / num_batches
        avg_hsic = total_hsic / num_batches
        avg_neyman = total_neyman / num_batches

        self.train_losses.append(avg_loss)
        self.epoch += 1

        # Validation if provided
        val_metrics = {}
        if val_loader is not None:
            val_metrics = self.validate(val_loader, verbose)
            self.val_losses.append(val_metrics.get("val_loss", float("inf")))

        metrics = {
            "train_loss": avg_loss,
            "train_mse": avg_mse,
            "train_prop": avg_prop,
            "train_hsic": avg_hsic,
            "train_neyman": avg_neyman,
            **val_metrics,
        }

        if verbose:
            logger.info(
                f"Epoch {self.epoch}: Train Loss={avg_loss:.4f}, "
                f"MSE={avg_mse:.4f}, Prop={avg_prop:.4f}, "
                f"HSIC={avg_hsic:.4f}, Neyman={avg_neyman:.4f}"
            )
            if val_metrics:
                logger.info(f"  Val Loss={val_metrics.get('val_loss', 0):.4f}")

        return metrics

    def validate(self, val_loader: DataLoader, verbose: bool = True) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_mse = 0.0
        total_prop = 0.0
        num_batches = 0

        with torch.no_grad():
            for x, t, y in val_loader:
                x, t, y = x.to(self.device), t.to(self.device), y.to(self.device)
                outputs = self.model(x)

                # Compute validation losses
                g = outputs["g"]
                m = outputs["m"]

                prop_loss = self.bce_loss(g, t)
                mse_loss = self.mse_loss(m, y)
                loss = mse_loss + prop_loss

                total_loss += loss.item()
                total_mse += mse_loss.item()
                total_prop += prop_loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        avg_mse = total_mse / num_batches
        avg_prop = total_prop / num_batches

        metrics = {
            "val_loss": avg_loss,
            "val_mse": avg_mse,
            "val_prop": avg_prop,
        }

        return metrics

    def save_checkpoint(self, path: Path, save_optimizer: bool = True):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict() if save_optimizer else None,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "hsic_weight": self.hsic_weight,
            "neyman_weight": self.neyman_weight,
            "uncertainty_beta": self.uncertainty_beta,
        }
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: Path, load_optimizer: bool = True):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if load_optimizer and checkpoint["optimizer_state_dict"] is not None:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.train_losses = checkpoint["train_losses"]
        self.val_losses = checkpoint["val_losses"]
        self.best_val_loss = checkpoint["best_val_loss"]
        logger.info(f"Checkpoint loaded from {path}, epoch {self.epoch}")


def prepare_dataloaders(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    train_ratio: float = 0.8,
    batch_size: int = 64,
    random_seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """Prepare training and validation dataloaders."""
    n_samples = X.shape[0]
    n_train = int(n_samples * train_ratio)

    # Shuffle indices
    indices = np.random.RandomState(random_seed).permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    # Convert to tensors
    X_tensor = torch.FloatTensor(X)
    T_tensor = torch.FloatTensor(T)
    Y_tensor = torch.FloatTensor(Y)

    # Create datasets
    train_dataset = TensorDataset(
        X_tensor[train_indices], T_tensor[train_indices], Y_tensor[train_indices]
    )
    val_dataset = TensorDataset(
        X_tensor[val_indices], T_tensor[val_indices], Y_tensor[val_indices]
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")

    return train_loader, val_loader


def train(
    X: np.ndarray,
    T: np.ndarray,
    Y: np.ndarray,
    input_dim: int,
    epochs: int = 100,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    hsic_weight: float = 0.1,
    neyman_weight: float = 1.0,
    use_pyro: bool = True,
    use_uncertainty_weighting: bool = True,
    output_dir: Path = Path("output"),
    checkpoint_freq: int = 10,
) -> OptimizedDragonNet:
    """Main training function."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare data
    train_loader, val_loader = prepare_dataloaders(
        X, T, Y, train_ratio=0.8, batch_size=batch_size
    )

    # Initialize model
    model = OptimizedDragonNet(
        input_dim=input_dim,
        shared_widths=(200, 200),
        split_sizes=(100, 100),
        hidden_tau=64,
        dropout=0.1,
        use_pyro=use_pyro,
        dp_k=10,
        uncertainty_dropout=True,
    )

    # Initialize trainer
    trainer = DragonNetTrainer(
        model=model,
        learning_rate=learning_rate,
        hsic_weight=hsic_weight,
        neyman_weight=neyman_weight,
        use_uncertainty_weighting=use_uncertainty_weighting,
    )

    # Training loop
    best_model_path = output_dir / "best_model.pt"
    for epoch in range(epochs):
        metrics = trainer.train_epoch(train_loader, val_loader, verbose=True)

        # Save best model
        if "val_loss" in metrics and metrics["val_loss"] < trainer.best_val_loss:
            trainer.best_val_loss = metrics["val_loss"]
            trainer.save_checkpoint(best_model_path, save_optimizer=False)

        # Save periodic checkpoint
        if checkpoint_freq > 0 and (epoch + 1) % checkpoint_freq == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            trainer.save_checkpoint(checkpoint_path)

    # Load best model
    if best_model_path.exists():
        trainer.load_checkpoint(best_model_path, load_optimizer=False)

    return trainer.model


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(
        description="Train Optimized DragonNet for causal inference"
    )
    parser.add_argument(
        "--data-dir", type=str, required=True, help="Directory containing data files"
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="Number of training epochs"
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument(
        "--hsic-weight", type=float, default=0.1, help="Weight for HSIC loss"
    )
    parser.add_argument(
        "--neyman-weight", type=float, default=1.0, help="Weight for Neyman loss"
    )
    parser.add_argument(
        "--use-pyro", action="store_true", default=True, help="Use Pyro noise model"
    )
    parser.add_argument(
        "--no-uncertainty-weighting",
        action="store_false",
        dest="use_uncertainty_weighting",
        default=True,
        help="Disable uncertainty-based weighting",
    )
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory"
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=10, help="Checkpoint frequency (epochs)"
    )

    args = parser.parse_args()

    # Example: Load your data here
    # This is a placeholder - you need to implement your data loading logic
    logger.info(f"Loading data from {args.data_dir}")
    # X, T, Y = load_your_data(args.data_dir)

    # For now, create synthetic data for demonstration
    n_samples = 1000
    input_dim = 50
    X = np.random.randn(n_samples, input_dim)
    T = np.random.binomial(1, 0.5, n_samples).astype(np.float32)
    Y = np.random.randn(n_samples).astype(np.float32)

    logger.info(f"Training with {n_samples} samples, input_dim={input_dim}")

    model = train(
        X=X,
        T=T,
        Y=Y,
        input_dim=input_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hsic_weight=args.hsic_weight,
        neyman_weight=args.neyman_weight,
        use_pyro=args.use_pyro,
        use_uncertainty_weighting=args.use_uncertainty_weighting,
        output_dir=Path(args.output_dir),
        checkpoint_freq=args.checkpoint_freq,
    )

    logger.info("Training completed!")
    logger.info(f"Model saved to {args.output_dir}/best_model.pt")


if __name__ == "__main__":
    main()