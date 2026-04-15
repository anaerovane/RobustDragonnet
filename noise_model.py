import torch
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal
from pyro.optim import Adam
import numpy as np


class TruncatedDPGMM:
    """Truncated stick-breaking DP Gaussian mixture implemented in Pyro.

    Practical truncated DP approximation with K components (configurable).
    Provides SVI fit and posterior predictive mean for residual de-noising.
    """

    def __init__(self, K=10, alpha=1.0, init_scale=1.0, device='cpu'):
        self.K = K
        self.alpha = alpha
        self.device = device
        self._fitted = False
        self.init_scale = init_scale

    def model(self, x):
        N = x.shape[0]
        K = self.K
        alpha = self.alpha
        with pyro.plate("components", K):
            v = pyro.sample("v", dist.Beta(torch.ones(K, device=self.device), alpha * torch.ones(K, device=self.device)))
            mu = pyro.sample("mu", dist.Normal(torch.zeros(K, device=self.device), self.init_scale * torch.ones(K, device=self.device)))
            sigma = pyro.sample("sigma", dist.LogNormal(-1.0 * torch.ones(K, device=self.device), 0.5 * torch.ones(K, device=self.device)))
        # stick breaking to weights
        stick_rem = torch.cumprod(1.0 - v + 1e-8, dim=0)
        w = torch.empty(K, device=self.device)
        w[0] = v[0]
        w[1:] = v[1:] * stick_rem[:-1]
        # construct mixture distribution to avoid discrete latent sampling (enumeration issues)
        with pyro.plate("data", N):
            mix = dist.Categorical(probs=w)
            comp = dist.Normal(loc=mu, scale=sigma)
            # use MixtureSameFamily to represent marginal over z
            mixture = dist.MixtureSameFamily(mix, comp)
            pyro.sample("obs", mixture, obs=x)

    def fit(self, x, num_steps=200, lr=1e-2, verbose=False):
        pyro.clear_param_store()
        guide = AutoDiagonalNormal(self.model)
        optim = Adam({"lr": lr})
        svi = SVI(self.model, guide, optim, loss=Trace_ELBO())
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        for i in range(num_steps):
            loss = svi.step(x_tensor)
            if verbose and (i % max(1, num_steps // 10) == 0):
                print(f"SVI step {i}/{num_steps} loss={loss:.4f}")
        self._guide = guide
        self._svi = svi
        self._optim = optim
        self._fitted = True

    def partial_fit(self, x_batch, num_steps=10, lr=1e-2, verbose=False):
        """Perform a few SVI update steps using the provided residual batch (online update).

        Keeps a persistent guide and SVI instance across calls for streaming updates.
        Returns the last SVI loss.
        """
        if not hasattr(self, '_guide') or not self._fitted:
            # initialize if not
            pyro.clear_param_store()
            self._guide = AutoDiagonalNormal(self.model)
            self._optim = Adam({"lr": lr})
            self._svi = SVI(self.model, self._guide, self._optim, loss=Trace_ELBO())
            self._fitted = True

        x_tensor = torch.tensor(x_batch, dtype=torch.float32, device=self.device)
        last_loss = None
        for i in range(num_steps):
            last_loss = self._svi.step(x_tensor)
            if verbose and (i % max(1, num_steps // 10) == 0):
                print(f"partial SVI step {i}/{num_steps} loss={last_loss:.4f}")
        return last_loss

    def posterior_predictive_mean(self, x, num_samples=50):
        """Return posterior predictive mean per data point using guide samples.

        Uses Pyro's Predictive to sample latent parameters from the variational guide
        and compute the mixture predictive mean (sum_k w_k * mu_k) for each sample.
        """
        if not self._fitted:
            raise RuntimeError("DPGMM not fitted. Call fit() or partial_fit() first.")
        x_tensor = torch.tensor(x, dtype=torch.float32, device=self.device)
        from pyro.infer import Predictive
        predictive = Predictive(self.model, guide=self._guide, num_samples=num_samples, return_sites=["v", "mu", "sigma"]) 
        samples = predictive(x_tensor)
        v_samples = samples['v']  # (num_samples, K)
        mu_samples = samples['mu']  # (num_samples, K)
        # compute predictive mean per sample (scalar)
        sample_means = []
        for i in range(v_samples.shape[0]):
            v = v_samples[i]
            mu = mu_samples[i]
            stick_rem = torch.cumprod(1.0 - v + 1e-8, dim=0)
            w = torch.empty_like(v)
            w[0] = v[0]
            w[1:] = v[1:] * stick_rem[:-1]
            sample_means.append((w * mu).sum().item())
        # predictive mean is same across data points (mixture mean is global); broadcast to data size
        mean_scalar = float(np.mean(sample_means))
        return np.full((x_tensor.shape[0],), mean_scalar, dtype=np.float32)
