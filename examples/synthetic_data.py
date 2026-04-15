#!/usr/bin/env python3
"""
Example: Training Optimized DragonNet on synthetic data.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimized_dragonnet import OptimizedDragonNet
from src.optimized_dragonnet.train import train, DragonNetTrainer, prepare_dataloaders


def generate_synthetic_data(n_samples=5000, input_dim=20, treatment_effect=2.0, noise_level=0.5, seed=42):
    """Generate synthetic observational data with known treatment effects.

    Data generating process:
    - Covariates: X ~ N(0, I)
    - Propensity: p(T=1|X) = sigmoid(0.3 * X[:,0] + 0.2 * X[:,1] - 0.1)
    - Baseline: Y(0) = 2 * X[:,0] + 1.5 * X[:,1] + N(0, noise_level)
    - Treatment effect: τ(X) = treatment_effect * sigmoid(0.2 * X[:,2])
    - Observed outcome: Y = Y(0) + T * τ(X)

    Args:
        n_samples: Number of samples
        input_dim: Dimension of covariates
        treatment_effect: Scale of treatment effect
        noise_level: Standard deviation of outcome noise
        seed: Random seed

    Returns:
        X: (n_samples, input_dim) covariates
        T: (n_samples,) treatment assignments
        Y: (n_samples,) observed outcomes
        tau_true: (n_samples,) true treatment effects
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate covariates
    X = np.random.randn(n_samples, input_dim)

    # Propensity score (non-linear)
    propensity_logit = 0.3 * X[:, 0] + 0.2 * X[:, 1] - 0.1
    propensity = 1 / (1 + np.exp(-propensity_logit))
    T = np.random.binomial(1, propensity).astype(np.float32)

    # Baseline outcome under control
    Y0 = 2.0 * X[:, 0] + 1.5 * X[:, 1] + noise_level * np.random.randn(n_samples)

    # Heterogeneous treatment effect
    tau_true = treatment_effect * (1 / (1 + np.exp(-0.2 * X[:, 2])))

    # Observed outcome
    Y = Y0 + T * tau_true + noise_level * 0.5 * np.random.randn(n_samples)

    return X.astype(np.float32), T, Y.astype(np.float32), tau_true.astype(np.float32)


def evaluate_predictions(model, X_test, T_test, Y_test, tau_true_test, device="cpu"):
    """Evaluate model predictions on test data."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X_test).to(device)
        outputs = model(X_tensor, return_uncertainty=True, mc_samples=30)

        # Extract predictions
        tau_pred = outputs["tau"].cpu().numpy()
        tau_var = outputs["tau_var"].cpu().numpy() if "tau_var" in outputs else None
        g_pred = outputs["g"].cpu().numpy()
        m_pred = outputs["m"].cpu().numpy()

    # Calculate metrics
    tau_mae = np.mean(np.abs(tau_pred - tau_true_test))
    tau_rmse = np.sqrt(np.mean((tau_pred - tau_true_test) ** 2))
    propensity_auc = np.mean((g_pred > 0.5) == T_test)
    mse_baseline = np.mean((m_pred - Y_test) ** 2)

    print(f"Treatment Effect Prediction:")
    print(f"  MAE: {tau_mae:.4f}")
    print(f"  RMSE: {tau_rmse:.4f}")
    print(f"Propensity AUC: {propensity_auc:.4f}")
    print(f"Baseline MSE: {mse_baseline:.4f}")

    if tau_var is not None:
        avg_uncertainty = np.mean(tau_var)
        print(f"Average uncertainty (variance): {avg_uncertainty:.4f}")

    return {
        "tau_pred": tau_pred,
        "tau_true": tau_true_test,
        "tau_var": tau_var,
        "tau_mae": tau_mae,
        "tau_rmse": tau_rmse,
        "propensity_auc": propensity_auc,
        "mse_baseline": mse_baseline,
    }


def plot_results(results, output_dir="output"):
    """Plot evaluation results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    tau_pred = results["tau_pred"]
    tau_true = results["tau_true"]
    tau_var = results["tau_var"]

    # Plot 1: True vs Predicted Treatment Effects
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.scatter(tau_true, tau_pred, alpha=0.5, s=10)
    plt.plot([tau_true.min(), tau_true.max()], [tau_true.min(), tau_true.max()], "r--", label="Perfect")
    plt.xlabel("True Treatment Effect")
    plt.ylabel("Predicted Treatment Effect")
    plt.title("True vs Predicted Treatment Effects")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 2: Uncertainty vs Prediction Error
    if tau_var is not None:
        plt.subplot(1, 2, 2)
        prediction_error = np.abs(tau_pred - tau_true)
        plt.scatter(tau_var, prediction_error, alpha=0.5, s=10)
        plt.xlabel("Prediction Variance (Uncertainty)")
        plt.ylabel("Absolute Prediction Error")
        plt.title("Uncertainty vs Prediction Error")
        plt.grid(True, alpha=0.3)

        # Calculate correlation
        corr = np.corrcoef(tau_var, prediction_error)[0, 1]
        plt.text(0.05, 0.95, f"Correlation: {corr:.3f}",
                transform=plt.gca().transAxes, verticalalignment="top")

    plt.tight_layout()
    plt.savefig(output_dir / "synthetic_evaluation.png", dpi=150)
    plt.close()

    print(f"Plots saved to {output_dir / 'synthetic_evaluation.png'}")


def main():
    """Main example script."""
    print("=" * 60)
    print("Optimized DragonNet - Synthetic Data Example")
    print("=" * 60)

    # Configuration
    n_samples = 5000
    input_dim = 20
    treatment_effect = 2.0
    noise_level = 1.0
    test_ratio = 0.2
    output_dir = Path("output/synthetic_example")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data
    print("\n1. Generating synthetic data...")
    X, T, Y, tau_true = generate_synthetic_data(
        n_samples=n_samples,
        input_dim=input_dim,
        treatment_effect=treatment_effect,
        noise_level=noise_level,
        seed=42
    )

    # Split into train/test
    n_test = int(n_samples * test_ratio)
    X_train, X_test = X[n_test:], X[:n_test]
    T_train, T_test = T[n_test:], T[:n_test]
    Y_train, Y_test = Y[n_test:], Y[:n_test]
    tau_true_train, tau_true_test = tau_true[n_test:], tau_true[:n_test]

    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Input dimension: {input_dim}")
    print(f"   Treatment prevalence: {T_train.mean():.3f}")

    # Train model
    print("\n2. Training Optimized DragonNet...")
    model = train(
        X=X_train,
        T=T_train,
        Y=Y_train,
        input_dim=input_dim,
        epochs=50,
        batch_size=64,
        learning_rate=1e-3,
        hsic_weight=0.1,
        neyman_weight=1.0,
        use_pyro=True,
        use_uncertainty_weighting=True,
        output_dir=output_dir / "training",
        checkpoint_freq=10,
    )

    # Evaluate
    print("\n3. Evaluating on test data...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    results = evaluate_predictions(
        model, X_test, T_test, Y_test, tau_true_test, device=device
    )

    # Plot results
    print("\n4. Generating plots...")
    plot_results(results, output_dir=output_dir)

    # Example of uncertainty-based weighting
    print("\n5. Demonstrating uncertainty-based weighting...")
    if results["tau_var"] is not None:
        from src.optimized_dragonnet.model import uncertainty_based_weights

        variances = torch.FloatTensor(results["tau_var"])
        weights = uncertainty_based_weights(
            variances, beta=1.0, min_weight=0.1, max_weight=1.0
        )

        print(f"   Sample weights range: [{weights.min():.3f}, {weights.max():.3f}]")
        print(f"   Mean weight: {weights.mean():.3f}")

        # Show correlation between weight and error
        errors = np.abs(results["tau_pred"] - results["tau_true"])
        weight_corr = np.corrcoef(weights.numpy(), errors)[0, 1]
        print(f"   Correlation between weight and error: {weight_corr:.3f}")
        print("   (Negative correlation indicates that high-error samples get lower weights)")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()