#!/usr/bin/env python3
"""
Example: Analyzing prediction uncertainties in Optimized DragonNet.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from scipy import stats

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.optimized_dragonnet import OptimizedDragonNet
from src.optimized_dragonnet.train import DragonNetTrainer, prepare_dataloaders


def generate_data_with_varying_noise(n_samples=3000, input_dim=30, seed=42):
    """Generate data with regions of different noise levels.

    Creates three regions:
    1. Low noise: Clear treatment effect signal
    2. Medium noise: Moderate uncertainty
    3. High noise: High uncertainty, ambiguous treatment effect

    Args:
        n_samples: Total number of samples
        input_dim: Dimension of covariates
        seed: Random seed

    Returns:
        X: Covariates with region indicator in first dimension
        T: Treatment assignments
        Y: Observed outcomes with varying noise
        region_labels: 0=low, 1=medium, 2=high noise
        tau_true: True treatment effects
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Generate covariates
    X = np.random.randn(n_samples, input_dim)

    # Create three regions based on X[:, 0]
    region_probs = np.array([0.4, 0.35, 0.25])  # Low, medium, high noise regions
    region_cumsum = np.cumsum(region_probs)
    u = np.random.rand(n_samples)
    region_labels = np.digitize(u, region_cumsum)

    # Region-specific parameters
    noise_levels = [0.3, 0.8, 1.5]  # Low, medium, high noise
    effect_strengths = [2.0, 1.5, 0.8]  # Strong, moderate, weak effects

    # Propensity score (region-dependent)
    propensity_logit = 0.3 * X[:, 0] + 0.2 * X[:, 1] - 0.1
    propensity = 1 / (1 + np.exp(-propensity_logit))
    T = np.random.binomial(1, propensity).astype(np.float32)

    # Generate outcomes with region-specific noise and effects
    Y = np.zeros(n_samples)
    tau_true = np.zeros(n_samples)

    for region in range(3):
        mask = region_labels == region
        n_region = mask.sum()

        if n_region == 0:
            continue

        # Baseline outcome
        Y0_region = 2.0 * X[mask, 0] + 1.5 * X[mask, 1]

        # Treatment effect
        tau_region = effect_strengths[region] * (1 / (1 + np.exp(-0.3 * X[mask, 2])))

        # Add noise
        noise = noise_levels[region] * np.random.randn(n_region)

        # Observed outcome
        Y_region = Y0_region + T[mask] * tau_region + noise

        Y[mask] = Y_region
        tau_true[mask] = tau_region

    return (
        X.astype(np.float32),
        T,
        Y.astype(np.float32),
        region_labels,
        tau_true.astype(np.float32),
    )


def analyze_uncertainty_calibration(model, X, tau_true, region_labels, device="cpu"):
    """Analyze how well uncertainty estimates correlate with prediction error."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        outputs = model(X_tensor, return_uncertainty=True, mc_samples=50)

        tau_pred = outputs["tau"].cpu().numpy()
        tau_var = outputs["tau_var"].cpu().numpy() if "tau_var" in outputs else None

    if tau_var is None:
        print("No uncertainty estimates available.")
        return None

    # Calculate prediction errors
    errors = np.abs(tau_pred - tau_true)
    std_uncertainty = np.sqrt(tau_var)  # Convert variance to standard deviation

    # Overall correlation
    overall_corr = np.corrcoef(std_uncertainty, errors)[0, 1]

    print("=" * 60)
    print("Uncertainty Calibration Analysis")
    print("=" * 60)
    print(f"Overall correlation (uncertainty vs error): {overall_corr:.4f}")
    print()

    # Analyze by region
    regions = ["Low Noise", "Medium Noise", "High Noise"]
    results = {}

    for region_id, region_name in enumerate(regions):
        mask = region_labels == region_id
        if mask.sum() == 0:
            continue

        region_errors = errors[mask]
        region_uncertainty = std_uncertainty[mask]

        region_corr = np.corrcoef(region_uncertainty, region_errors)[0, 1]
        mean_error = region_errors.mean()
        mean_uncertainty = region_uncertainty.mean()
        calibration_ratio = mean_uncertainty / (mean_error + 1e-8)

        results[region_name] = {
            "n_samples": mask.sum(),
            "mean_error": mean_error,
            "mean_uncertainty": mean_uncertainty,
            "correlation": region_corr,
            "calibration_ratio": calibration_ratio,
        }

        print(f"{region_name} Region (n={mask.sum()}):")
        print(f"  Mean prediction error: {mean_error:.4f}")
        print(f"  Mean uncertainty (std): {mean_uncertainty:.4f}")
        print(f"  Uncertainty/Error ratio: {calibration_ratio:.4f}")
        print(f"  Correlation (uncertainty vs error): {region_corr:.4f}")
        print()

    return {
        "errors": errors,
        "std_uncertainty": std_uncertainty,
        "overall_corr": overall_corr,
        "region_results": results,
        "region_labels": region_labels,
        "tau_pred": tau_pred,
        "tau_true": tau_true,
    }


def plot_uncertainty_analysis(results, output_dir="output"):
    """Create comprehensive uncertainty analysis plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    errors = results["errors"]
    std_uncertainty = results["std_uncertainty"]
    region_labels = results["region_labels"]
    tau_pred = results["tau_pred"]
    tau_true = results["tau_true"]

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(15, 10))

    # 1. Uncertainty vs Error scatter plot
    ax1 = plt.subplot(2, 3, 1)
    scatter = ax1.scatter(std_uncertainty, errors, alpha=0.6, s=20, c=region_labels, cmap="viridis")
    ax1.set_xlabel("Prediction Uncertainty (Standard Deviation)")
    ax1.set_ylabel("Absolute Prediction Error")
    ax1.set_title("Uncertainty vs Prediction Error")
    ax1.grid(True, alpha=0.3)

    # Add correlation text
    corr = results["overall_corr"]
    ax1.text(0.05, 0.95, f"Correlation: {corr:.3f}",
            transform=ax1.transAxes, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # 2. Calibration plot (reliability diagram)
    ax2 = plt.subplot(2, 3, 2)
    # Bin by uncertainty and compute average error in each bin
    n_bins = 10
    uncertainty_bins = np.percentile(std_uncertainty, np.linspace(0, 100, n_bins + 1))
    bin_indices = np.digitize(std_uncertainty, uncertainty_bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_errors = []
    bin_uncertainties = []
    bin_counts = []

    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if mask.sum() > 0:
            bin_errors.append(errors[mask].mean())
            bin_uncertainties.append(std_uncertainty[mask].mean())
            bin_counts.append(mask.sum())

    ax2.plot(bin_uncertainties, bin_errors, "o-", linewidth=2, markersize=8)
    ax2.plot([0, max(bin_uncertainties)], [0, max(bin_uncertainties)], "r--", alpha=0.7, label="Perfect calibration")
    ax2.set_xlabel("Average Uncertainty (Std)")
    ax2.set_ylabel("Average Error")
    ax2.set_title("Calibration: Uncertainty vs Actual Error")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Region comparison
    ax3 = plt.subplot(2, 3, 3)
    regions = ["Low Noise", "Medium Noise", "High Noise"]
    region_results = results["region_results"]

    metrics = ["mean_error", "mean_uncertainty", "correlation"]
    metric_names = ["Mean Error", "Mean Uncertainty", "Correlation"]
    colors = ["skyblue", "lightgreen", "lightcoral"]

    x = np.arange(len(regions))
    width = 0.25

    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        values = [region_results.get(r, {}).get(metric, 0) for r in regions]
        ax3.bar(x + i * width - width, values, width, label=metric_name, color=colors[i])

    ax3.set_xlabel("Noise Region")
    ax3.set_ylabel("Value")
    ax3.set_title("Metrics by Noise Region")
    ax3.set_xticks(x)
    ax3.set_xticklabels(regions, rotation=45)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis="y")

    # 4. True vs Predicted with uncertainty bands
    ax4 = plt.subplot(2, 3, 4)
    # Sort by true treatment effect for cleaner plot
    sort_idx = np.argsort(tau_true)
    tau_true_sorted = tau_true[sort_idx]
    tau_pred_sorted = tau_pred[sort_idx]
    uncertainty_sorted = std_uncertainty[sort_idx]

    ax4.plot(tau_true_sorted, tau_pred_sorted, "b-", alpha=0.7, linewidth=1)
    ax4.fill_between(
        tau_true_sorted,
        tau_pred_sorted - uncertainty_sorted,
        tau_pred_sorted + uncertainty_sorted,
        alpha=0.3,
        color="blue",
        label="±1 std uncertainty"
    )
    ax4.plot([tau_true.min(), tau_true.max()], [tau_true.min(), tau_true.max()], "r--", alpha=0.5, label="Perfect")
    ax4.set_xlabel("True Treatment Effect")
    ax4.set_ylabel("Predicted Treatment Effect")
    ax4.set_title("Predictions with Uncertainty Bands")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. Error distribution by uncertainty quartile
    ax5 = plt.subplot(2, 3, 5)
    uncertainty_quartiles = np.percentile(std_uncertainty, [25, 50, 75])
    quartile_labels = ["Q1 (Low)", "Q2", "Q3", "Q4 (High)"]
    quartile_errors = []

    for q in range(4):
        if q == 0:
            mask = std_uncertainty <= uncertainty_quartiles[0]
        elif q == 1:
            mask = (std_uncertainty > uncertainty_quartiles[0]) & (std_uncertainty <= uncertainty_quartiles[1])
        elif q == 2:
            mask = (std_uncertainty > uncertainty_quartiles[1]) & (std_uncertainty <= uncertainty_quartiles[2])
        else:
            mask = std_uncertainty > uncertainty_quartiles[2]

        quartile_errors.append(errors[mask])

    bp = ax5.boxplot(quartile_errors, labels=quartile_labels, patch_artist=True)
    for patch, color in zip(bp["boxes"], ["lightgreen", "lightblue", "lightyellow", "lightcoral"]):
        patch.set_facecolor(color)

    ax5.set_xlabel("Uncertainty Quartile")
    ax5.set_ylabel("Prediction Error")
    ax5.set_title("Error Distribution by Uncertainty Level")
    ax5.grid(True, alpha=0.3, axis="y")

    # 6. Uncertainty histogram
    ax6 = plt.subplot(2, 3, 6)
    ax6.hist(std_uncertainty, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
    ax6.set_xlabel("Prediction Uncertainty (Std)")
    ax6.set_ylabel("Frequency")
    ax6.set_title("Distribution of Prediction Uncertainties")
    ax6.grid(True, alpha=0.3)

    plt.suptitle("Optimized DragonNet - Uncertainty Analysis", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / "uncertainty_analysis.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\nAnalysis plots saved to {output_dir / 'uncertainty_analysis.png'}")


def train_model_for_analysis(X_train, T_train, Y_train, input_dim, output_dir):
    """Train a model for uncertainty analysis."""
    from src.optimized_dragonnet.train import train

    model = train(
        X=X_train,
        T=T_train,
        Y=Y_train,
        input_dim=input_dim,
        epochs=80,
        batch_size=64,
        learning_rate=1e-3,
        hsic_weight=0.1,
        neyman_weight=1.0,
        use_pyro=True,
        use_uncertainty_weighting=True,
        output_dir=output_dir / "training",
        checkpoint_freq=20,
    )

    return model


def main():
    """Main analysis script."""
    print("=" * 60)
    print("Optimized DragonNet - Uncertainty Analysis")
    print("=" * 60)

    # Configuration
    n_samples = 4000
    input_dim = 30
    test_ratio = 0.3
    output_dir = Path("output/uncertainty_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate data with varying noise regions
    print("\n1. Generating data with varying noise levels...")
    X, T, Y, region_labels, tau_true = generate_data_with_varying_noise(
        n_samples=n_samples, input_dim=input_dim, seed=42
    )

    # Split into train/test
    n_test = int(n_samples * test_ratio)
    X_train, X_test = X[n_test:], X[:n_test]
    T_train, T_test = T[n_test:], T[:n_test]
    Y_train, Y_test = Y[n_test:], Y[:n_test]
    region_labels_train, region_labels_test = region_labels[n_test:], region_labels[:n_test]
    tau_true_train, tau_true_test = tau_true[n_test:], tau_true[:n_test]

    print(f"   Training samples: {len(X_train)}")
    print(f"   Test samples: {len(X_test)}")
    print(f"   Region distribution (train):")
    for region_id, region_name in enumerate(["Low Noise", "Medium Noise", "High Noise"]):
        count = (region_labels_train == region_id).sum()
        print(f"     {region_name}: {count} samples ({count/len(X_train)*100:.1f}%)")

    # Train model
    print("\n2. Training model with uncertainty estimation...")
    model = train_model_for_analysis(
        X_train, T_train, Y_train, input_dim, output_dir
    )

    # Analyze uncertainties
    print("\n3. Analyzing prediction uncertainties...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    results = analyze_uncertainty_calibration(
        model, X_test, tau_true_test, region_labels_test, device=device
    )

    if results is not None:
        # Generate plots
        print("\n4. Generating analysis plots...")
        plot_uncertainty_analysis(results, output_dir=output_dir)

        # Save detailed results
        np.savez(
            output_dir / "uncertainty_results.npz",
            errors=results["errors"],
            std_uncertainty=results["std_uncertainty"],
            tau_pred=results["tau_pred"],
            tau_true=results["tau_true"],
            region_labels=results["region_labels"],
            overall_corr=results["overall_corr"],
        )

        print(f"\nDetailed results saved to {output_dir / 'uncertainty_results.npz'}")

    print("\n" + "=" * 60)
    print("Analysis completed!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()