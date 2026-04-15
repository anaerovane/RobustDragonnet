import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import pyro
    from .noise_model import TruncatedDPGMM
    _PYRO_AVAILABLE = True
except Exception:
    _PYRO_AVAILABLE = False


def uncertainty_based_weights(variances, beta=1.0, min_weight=0.1, max_weight=1.0):
    """Compute sample weights based on prediction uncertainties.

    Weight = min_weight + (max_weight - min_weight) * exp(-beta * variance)

    Args:
        variances: tensor of shape (N,) containing prediction variances
        beta: temperature parameter controlling sensitivity to variance
        min_weight: minimum weight for high-uncertainty samples
        max_weight: maximum weight for low-uncertainty samples

    Returns:
        weights: tensor of shape (N,) with weights in [min_weight, max_weight]
    """
    weights = min_weight + (max_weight - min_weight) * torch.exp(-beta * variances)
    return weights


def rbf_kernel(x, sigma=None):
    """Compute RBF kernel matrix for input x (N x d)."""
    diff = x.unsqueeze(1) - x.unsqueeze(0)
    sq = (diff ** 2).sum(-1)
    if sigma is None:
        # median heuristic
        with torch.no_grad():
            median = torch.median(sq[sq > 0])
            sigma = torch.sqrt(0.5 * median + 1e-6)
    k = torch.exp(-sq / (2 * sigma ** 2 + 1e-8))
    return k


def hsic_loss(x, y, sigma_x=None, sigma_y=None):
    """Unbiased HSIC estimator (Gretton et al.) for batches.

    Minimizes statistical dependence between x and y.
    """
    # x, y: (N, d1), (N, d2)
    N = x.shape[0]
    K = rbf_kernel(x, sigma_x)
    L = rbf_kernel(y, sigma_y)
    H = torch.eye(N, device=x.device) - (1.0 / N) * torch.ones((N, N), device=x.device)
    # biased estimator: tr(KHLH) / (N-1)^2 ; here use simple centered form
    Kc = H @ K @ H
    Lc = H @ L @ H
    hsic = (Kc * Lc).sum() / ((N - 1) ** 2 + 1e-12)
    return hsic


def neyman_loss(y, t, m, g, tau):
    """Neyman-orthogonal loss: residual-on-residual MSE.

    L = E[ ( (Y - m(X)) - (T - g(X)) * tau(X) )^2 ]
    """
    r = y - m
    w = t - g
    return F.mse_loss(r, w * tau)


class OptimizedDragonNet(nn.Module):
    """Optimized DragonNet with HSIC, Neyman orthogonality and Pyro-based noise filtering.

    Features:
    - Shared representation, split into two subspaces; HSIC penalizes dependence.
    - Nuisance heads: propensity g(X), baseline m(X).
    - Target head: tau(X) (treatment effect) with Neyman orthogonal loss.
    - Optional Truncated-DP Gaussian mixture noise filter (Pyro) working on residuals.
    """

    def __init__(self, input_dim, shared_widths=(200, 200), split_sizes=(100, 100), hidden_tau=64, dropout=0.1, use_pyro=False, dp_k=10, uncertainty_dropout=True):
        super().__init__()
        self.input_dim = input_dim
        self.dropout_rate = dropout
        self.uncertainty_dropout = uncertainty_dropout

        layers = []
        prev = input_dim
        for w in shared_widths:
            layers.append(nn.Linear(prev, w))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = w
        self.shared = nn.Sequential(*layers)

        # split into two subvectors for HSIC independence constraint
        total = sum(split_sizes)
        if total != prev:
            # linear projection to match split sizes
            self.project = nn.Linear(prev, total)
            shared_out_dim = total
        else:
            self.project = None
            shared_out_dim = prev

        # define splits
        self.split_sizes = split_sizes

        # propensity head
        self.propensity = nn.Sequential(nn.Linear(shared_out_dim, 1), nn.Sigmoid())

        # baseline outcome head m(X)
        self.baseline = nn.Sequential(nn.Linear(shared_out_dim, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, 1))

        # tau head for treatment effect - with dropout for uncertainty estimation
        tau_layers = []
        tau_layers.append(nn.Linear(shared_out_dim + 1, hidden_tau))
        tau_layers.append(nn.ReLU())
        if uncertainty_dropout:
            tau_layers.append(nn.Dropout(dropout))
        tau_layers.append(nn.Linear(hidden_tau, 1))
        self.tau_head = nn.Sequential(*tau_layers)

        # pyro noise model
        self.use_pyro = use_pyro and _PYRO_AVAILABLE
        if use_pyro and not _PYRO_AVAILABLE:
            raise RuntimeError("Pyro not available; install pyro-ppl to enable Bayesian noise filtering.")
        if self.use_pyro:
            self.noise_model = TruncatedDPGMM(K=dp_k)
        else:
            self.noise_model = None

    def split_repr(self, phi):
        if self.project is not None:
            phi = self.project(phi)
        splits = torch.split(phi, self.split_sizes, dim=1)
        return splits

    def forward(self, x, denoise=False, denoise_steps=50, return_uncertainty=False, mc_samples=20):
        """Forward pass with optional uncertainty estimation via MC Dropout.

        Args:
            x: input features
            denoise: whether to apply noise correction (requires use_pyro=True)
            denoise_steps: steps for noise model fitting
            return_uncertainty: if True, returns tau_mean and tau_var instead of single tau
            mc_samples: number of MC samples for uncertainty estimation

        Returns:
            dict with model outputs, including tau (or tau_mean/tau_var if return_uncertainty)
        """
        phi = self.shared(x)
        if self.project is not None:
            phi_proj = self.project(phi)
        else:
            phi_proj = phi
        a, b = self.split_repr(phi_proj)
        # propensity
        g = self.propensity(phi_proj).squeeze(-1)
        # baseline
        m = self.baseline(phi_proj).squeeze(-1)

        # tau uses phi + g as input
        tau_in = torch.cat([phi_proj, g.unsqueeze(-1)], dim=1)

        if return_uncertainty and self.uncertainty_dropout:
            # MC Dropout for uncertainty estimation
            self.train()  # ensure dropout is active
            tau_samples = []
            for _ in range(mc_samples):
                tau_sample = self.tau_head(tau_in).squeeze(-1)
                tau_samples.append(tau_sample)
            tau_samples = torch.stack(tau_samples, dim=0)  # (mc_samples, batch_size)
            tau_mean = tau_samples.mean(dim=0)
            tau_var = tau_samples.var(dim=0)
            tau = tau_mean
        else:
            # standard forward
            if self.training:
                self.tau_head.train()
            else:
                self.tau_head.eval()
            tau = self.tau_head(tau_in).squeeze(-1)
            tau_mean = tau
            tau_var = None

        # optional Pyro denoising: apply to residuals when requested
        noise_correction = None
        if denoise and self.use_pyro:
            # fit to residuals y - m externally via noise_model.partial_fit in training loop
            # this runtime hook is intentionally left empty — training loop will call
            # model.noise_model.partial_fit(resid_batch, ...)
            pass

        output_dict = dict(m=m, tau=tau, g=g, phi_a=a, phi_b=b, correction=noise_correction)
        if return_uncertainty and self.uncertainty_dropout:
            output_dict.update(tau_mean=tau_mean, tau_var=tau_var)

        return output_dict

    def denoise_residuals(self, resid_numpy, num_steps=200, lr=1e-2, verbose=False, return_variance=False):
        """Fit DP-GMM to residuals and return denoised residuals (posterior predictive mean and variance)."""
        if not self.use_pyro:
            raise RuntimeError("Pyro not enabled for this model.")
        self.noise_model.fit(resid_numpy, num_steps=num_steps, lr=lr, verbose=verbose)
        if return_variance:
            mean, var = self.noise_model.posterior_predictive_mean_var(resid_numpy)
            return mean, var
        else:
            return self.noise_model.posterior_predictive_mean(resid_numpy)


# Expose loss helpers at module level
__all__ = ["OptimizedDragonNet", "hsic_loss", "neyman_loss", "uncertainty_based_weights"]
