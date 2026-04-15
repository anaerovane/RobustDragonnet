#!/usr/bin/env python3
"""
Unit tests for Optimized DragonNet model.
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from optimized_dragonnet import (
    OptimizedDragonNet,
    hsic_loss,
    neyman_loss,
    uncertainty_based_weights,
)
from optimized_dragonnet.noise_model import TruncatedDPGMM


def test_model_initialization():
    """Test model initialization with different configurations."""
    # Basic initialization
    model = OptimizedDragonNet(input_dim=50)
    assert model.input_dim == 50
    assert model.use_pyro == False
    assert model.uncertainty_dropout == True

    # With Pyro enabled
    model = OptimizedDragonNet(input_dim=50, use_pyro=True)
    assert model.use_pyro == True
    assert hasattr(model, "noise_model") or model.noise_model is None

    # Custom architecture
    model = OptimizedDragonNet(
        input_dim=100,
        shared_widths=(128, 64),
        split_sizes=(32, 32),
        hidden_tau=32,
        dropout=0.2,
    )
    assert model.input_dim == 100


def test_forward_pass():
    """Test forward pass with and without uncertainty."""
    model = OptimizedDragonNet(input_dim=20)
    batch_size = 16

    # Create dummy input
    x = torch.randn(batch_size, 20)

    # Standard forward pass
    outputs = model(x)
    assert "tau" in outputs
    assert "g" in outputs
    assert "m" in outputs
    assert "phi_a" in outputs
    assert "phi_b" in outputs

    # Check shapes
    assert outputs["tau"].shape == (batch_size,)
    assert outputs["g"].shape == (batch_size,)
    assert outputs["m"].shape == (batch_size,)
    assert outputs["phi_a"].shape[0] == batch_size
    assert outputs["phi_b"].shape[0] == batch_size

    # Forward pass with uncertainty
    outputs = model(x, return_uncertainty=True, mc_samples=10)
    assert "tau_mean" in outputs
    assert "tau_var" in outputs
    assert outputs["tau_mean"].shape == (batch_size,)
    assert outputs["tau_var"].shape == (batch_size,)


def test_hsic_loss():
    """Test HSIC loss computation."""
    batch_size = 32
    dim1, dim2 = 10, 15

    x = torch.randn(batch_size, dim1)
    y = torch.randn(batch_size, dim2)

    loss = hsic_loss(x, y)
    assert loss.shape == ()
    assert loss.item() >= 0

    # HSIC should be symmetric
    loss_xy = hsic_loss(x, y)
    loss_yx = hsic_loss(y, x)
    assert torch.allclose(loss_xy, loss_yx, rtol=1e-4)


def test_neyman_loss():
    """Test Neyman orthogonal loss computation."""
    batch_size = 32

    y = torch.randn(batch_size)
    t = torch.randn(batch_size)
    m = torch.randn(batch_size)
    g = torch.randn(batch_size)
    tau = torch.randn(batch_size)

    loss = neyman_loss(y, t, m, g, tau)
    assert loss.shape == ()
    assert loss.item() >= 0


def test_uncertainty_based_weights():
    """Test uncertainty-based weight computation."""
    variances = torch.tensor([0.1, 0.5, 1.0, 2.0, 5.0])
    weights = uncertainty_based_weights(variances, beta=1.0)

    assert weights.shape == variances.shape
    assert torch.all(weights >= 0.1)  # min_weight
    assert torch.all(weights <= 1.0)  # max_weight

    # Higher variance should give lower weight
    assert weights[0] > weights[-1]  # var=0.1 vs var=5.0

    # Test with custom beta
    weights_beta2 = uncertainty_based_weights(variances, beta=2.0)
    weights_beta05 = uncertainty_based_weights(variances, beta=0.5)

    # Higher beta -> more sensitive to variance -> lower weights for high variance
    assert weights_beta2[-1] < weights_beta05[-1]


def test_noise_model_initialization():
    """Test TruncatedDPGMM initialization."""
    # Skip if Pyro not available
    try:
        import pyro
    except ImportError:
        pytest.skip("Pyro not available")

    model = TruncatedDPGMM(K=5, alpha=0.5, device="cpu")
    assert model.K == 5
    assert model.alpha == 0.5
    assert model.device == "cpu"
    assert not model._fitted


def test_noise_model_fit_predict():
    """Test DP-GMM fitting and prediction."""
    # Skip if Pyro not available
    try:
        import pyro
    except ImportError:
        pytest.skip("Pyro not available")

    # Generate synthetic residuals
    np.random.seed(42)
    n_samples = 100
    residuals = np.random.randn(n_samples).astype(np.float32)

    # Initialize and fit
    model = TruncatedDPGMM(K=3, device="cpu")
    model.fit(residuals, num_steps=50, verbose=False)

    # Test prediction
    mean = model.posterior_predictive_mean(residuals)
    var = model.posterior_predictive_variance(residuals)

    assert mean.shape == (n_samples,)
    assert var.shape == (n_samples,)
    assert np.all(var >= 0)

    # Test mean-var consistency
    mean2, var2 = model.posterior_predictive_mean_var(residuals)
    assert np.allclose(mean, mean2)
    assert np.allclose(var, var2)


def test_model_with_noise_model():
    """Test model integration with noise model."""
    # Skip if Pyro not available
    try:
        import pyro
    except ImportError:
        pytest.skip("Pyro not available")

    model = OptimizedDragonNet(input_dim=20, use_pyro=True, dp_k=3)
    batch_size = 16

    # Create dummy data
    x = torch.randn(batch_size, 20)
    y = torch.randn(batch_size)

    # Forward pass
    outputs = model(x)
    assert hasattr(model, "noise_model")
    assert model.noise_model is not None

    # Test denoising
    residuals = y - outputs["m"]
    residuals_np = residuals.detach().cpu().numpy()

    # Fit noise model
    denoised = model.denoise_residuals(residuals_np, num_steps=20, verbose=False)
    assert denoised.shape == residuals_np.shape

    # Test with variance
    denoised_mean, denoised_var = model.denoise_residuals(
        residuals_np, num_steps=20, verbose=False, return_variance=True
    )
    assert denoised_mean.shape == residuals_np.shape
    assert denoised_var.shape == residuals_np.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])