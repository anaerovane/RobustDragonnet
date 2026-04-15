# Optimized DragonNet

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.12+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Optimized DragonNet is a robust deep learning model for causal inference from complex observational data. It extends the original DragonNet architecture with three key improvements:

1. **HSIC Constraints**: Hilbert-Schmidt Independence Criterion (HSIC) regularization on the shared representation to reduce redundant information correlated with treatment assignment.
2. **Neyman Orthogonality**: Neyman-orthogonal loss function that reduces error propagation from propensity score estimation to outcome prediction.
3. **Uncertainty-Aware Weighting**: Bayesian uncertainty quantification with MC Dropout and uncertainty-based sample weighting to downweight high-uncertainty predictions.

This implementation corresponds to the thesis "面向复杂观测数据的抗噪声因果推断深度学习模型优化" (Noise-Robust Deep Learning Model Optimization for Causal Inference with Complex Observational Data).

## Features

- **HSIC-Regularized Representation**: Split representation with HSIC penalty to enforce independence between treatment-relevant and outcome-relevant features.
- **Neyman-Orthogonal Training**: Residual-on-residual loss for treatment effect estimation that is first-order insensitive to nuisance parameter errors.
- **Bayesian Uncertainty Quantification**:
  - MC Dropout for treatment effect uncertainty estimation
  - Dirichlet Process Gaussian Mixture Model (DP-GMM) for residual noise modeling
  - Uncertainty-based sample weighting during training
- **Pyro Integration**: Non-parametric Bayesian noise modeling using Pyro probabilistic programming.
- **Modular Design**: Clean, well-documented code with separate modules for model, training, and evaluation.

## Installation

### From Source

```bash
git clone https://github.com/username/optimized-dragonnet.git
cd optimized-dragonnet
pip install -e .
```

For development with additional dependencies:

```bash
pip install -e ".[dev,examples]"
```

### Dependencies

- Python 3.8+
- PyTorch >= 1.12.0
- Pyro-PPL >= 1.8.0
- NumPy >= 1.21.0
- SciPy >= 1.7.0

Optional dependencies for examples:
- pandas >= 1.5.0
- scikit-learn >= 1.2.0
- matplotlib >= 3.6.0
- seaborn >= 0.12.0

## Quick Start

### Basic Usage

```python
import torch
import numpy as np
from optimized_dragonnet import OptimizedDragonNet, hsic_loss, neyman_loss

# Create synthetic data
n_samples = 1000
input_dim = 50
X = np.random.randn(n_samples, input_dim)
T = np.random.binomial(1, 0.5, n_samples).astype(np.float32)
Y = np.random.randn(n_samples).astype(np.float32)

# Initialize model
model = OptimizedDragonNet(
    input_dim=input_dim,
    shared_widths=(200, 200),
    split_sizes=(100, 100),
    hidden_tau=64,
    dropout=0.1,
    use_pyro=True,  # Enable Bayesian noise modeling
    dp_k=10,        # Number of DP-GMM components
    uncertainty_dropout=True,  # Enable MC Dropout for uncertainty
)

# Forward pass
x_tensor = torch.FloatTensor(X[:10])
outputs = model(x_tensor, return_uncertainty=True, mc_samples=20)

print(f"Treatment effects: {outputs['tau'].detach().numpy()}")
print(f"Uncertainty (variance): {outputs['tau_var'].detach().numpy()}")
```

### Training

```python
from optimized_dragonnet.train import train

# Train model
trained_model = train(
    X=X,
    T=T,
    Y=Y,
    input_dim=input_dim,
    epochs=100,
    batch_size=64,
    learning_rate=1e-3,
    hsic_weight=0.1,
    neyman_weight=1.0,
    use_pyro=True,
    use_uncertainty_weighting=True,
    output_dir="output",
)
```

### Command Line Interface

```bash
# Train with default settings
python -m optimized_dragonnet.train \
    --data-dir ./data \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 1e-3 \
    --output-dir ./output

# Disable uncertainty weighting
python -m optimized_dragonnet.train \
    --data-dir ./data \
    --no-uncertainty-weighting \
    --output-dir ./output_no_uncertainty
```

## Model Architecture

### Core Components

1. **Shared Representation Network**: Multi-layer perceptron that extracts features from input covariates.
2. **Split Representation**: The shared representation is split into two sub-vectors for HSIC regularization.
3. **Nuisance Heads**:
   - Propensity score head `g(X)` for treatment assignment prediction
   - Baseline outcome head `m(X)` for outcome prediction under control
4. **Treatment Effect Head**: `τ(X)` head that takes both representation and propensity score as input.
5. **Uncertainty Estimation**: MC Dropout applied to the τ head for Bayesian uncertainty quantification.
6. **Noise Model**: Optional DP-GMM for residual noise modeling and denoising.

### Loss Functions

The total loss is a weighted combination of:

```
L = L_mse(m(X), Y) + L_bce(g(X), T) + λ_hsic * HSIC(φ_a, φ_b) + λ_neyman * L_neyman(Y, T, m, g, τ)
```

Where:
- `L_mse`: Mean squared error for baseline outcome
- `L_bce`: Binary cross-entropy for propensity score
- `HSIC`: Hilbert-Schmidt Independence Criterion between representation splits
- `L_neyman`: Neyman-orthogonal loss for treatment effect estimation

### Uncertainty-Based Weighting

During training, sample weights are computed based on prediction uncertainty:

```
weight = w_min + (w_max - w_min) * exp(-β * variance)
```

High-variance predictions (uncertain samples) receive lower weights, reducing their influence on parameter updates.

## API Reference

### Core Classes

#### `OptimizedDragonNet`

```python
model = OptimizedDragonNet(
    input_dim,                      # Dimension of input features
    shared_widths=(200, 200),       # Hidden layer sizes for shared network
    split_sizes=(100, 100),         # Sizes of representation splits for HSIC
    hidden_tau=64,                  # Hidden size for τ head
    dropout=0.1,                    # Dropout rate
    use_pyro=False,                 # Enable Pyro DP-GMM noise model
    dp_k=10,                        # Number of DP-GMM components
    uncertainty_dropout=True,       # Enable MC Dropout for uncertainty
)
```

**Methods:**
- `forward(x, denoise=False, return_uncertainty=False, mc_samples=20)`: Forward pass
- `denoise_residuals(resid_numpy, return_variance=False)`: Fit DP-GMM to residuals

#### `TruncatedDPGMM`

Bayesian non-parametric noise model using Dirichlet Process Gaussian Mixture.

```python
noise_model = TruncatedDPGMM(K=10, alpha=1.0, device='cpu')
noise_model.fit(residuals, num_steps=200)
mean, var = noise_model.posterior_predictive_mean_var(residuals)
```

### Loss Functions

- `hsic_loss(x, y)`: Hilbert-Schmidt Independence Criterion
- `neyman_loss(y, t, m, g, tau)`: Neyman-orthogonal loss
- `uncertainty_based_weights(variances, beta=1.0)`: Compute sample weights from uncertainties

### Training Utilities

The `train` module provides:

- `DragonNetTrainer`: Complete training loop with validation
- `train()`: High-level training function
- `prepare_dataloaders()`: Data preparation utilities

## Examples

See the `examples/` directory for complete examples:

- `examples/synthetic_data.py`: Training on synthetic data
- `examples/criteo_uplift.py`: Example with Criteo Uplift dataset format
- `examples/uncertainty_analysis.py`: Analyzing prediction uncertainties

Run examples with:

```bash
python examples/synthetic_data.py
```

## Experimental Results

The model was evaluated on six real-world datasets:

1. **EC-LIFT**: E-commerce lift modeling dataset
2. **MT-LIFT**: Multi-treatment lift dataset
3. **Criteo Uplift v2**: Large-scale advertising dataset
4. **Hillstrom**: Email marketing dataset
5. **LENTA**: News recommendation dataset
6. **US Census**: Socioeconomic dataset

Key findings:
- Improved Qini coefficient and AUUC compared to baseline methods
- More stable ranking performance under label noise
- Better robustness to selection bias and feature redundancy

## Citation

If you use this code in your research, please cite the associated thesis:

```
@thesis{dong2026optimized,
  title={面向复杂观测数据的抗噪声因果推断深度学习模型优化},
  author={董梧铮},
  year={2026},
  school={南开大学}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on the original DragonNet architecture: [Shi et al., 2019](https://arxiv.org/abs/1906.02120)
- HSIC implementation inspired by [Gretton et al., 2005](https://www.jmlr.org/papers/v6/gretton05a.html)
- Neyman orthogonality from [Chernozhukov et al., 2018](https://arxiv.org/abs/1608.00060)
- DP-GMM implementation using [Pyro](http://pyro.ai)