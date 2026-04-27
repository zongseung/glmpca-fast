"""PyTorch GPU backend for Poisson GLM-PCA (Townes et al. 2019).

Coordinate-block Newton on a CUDA device. Uses ``torch.linalg.solve`` to
batch the per-sample / per-feature ``(L, L)`` Newton systems. Designed
for L ≲ 32; for very large L the dense ``(N, L, L)`` Hessian materialises
and may dominate memory.
"""

from __future__ import annotations

import numpy as np


def _poisson_deviance(y, mu) -> float:
    import torch
    mu = mu.clamp(min=1e-9)
    log_term = torch.where(
        y > 0,
        y * (y / mu).log() - (y - mu),
        -(y - mu),
    )
    return float((2.0 * log_term).sum().clamp_min(0.0).item())


def fit_poisson_torch(
    y_np: np.ndarray,
    L: int,
    max_iter: int = 100,
    tol: float = 1e-4,
    penalty: float = 1.0,
    seed: int = 42,
    device: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)

    Y = torch.as_tensor(y_np, dtype=torch.float32, device=dev).t().contiguous()
    M, N = Y.shape
    Le = min(L, M, N)
    if Le < 2:
        raise ValueError(f"Need effective L >= 2; got {Le}")

    g = torch.Generator(device=dev).manual_seed(seed)
    Z = 0.05 * torch.randn(N, Le, generator=g, device=dev, dtype=torch.float32)
    V = 0.05 * torch.randn(M, Le, generator=g, device=dev, dtype=torch.float32)
    a = torch.log(Y.mean(dim=1) + 1.0).clamp(min=-10.0)

    eye_L = torch.eye(Le, device=dev, dtype=torch.float32)
    lam = max(float(penalty), 1e-6)

    deviance = []
    prev_dev = float("inf")
    n_iter = 0

    with torch.inference_mode():
        for it in range(max_iter):
            eta = a.unsqueeze(1) + V @ Z.t()
            mu = eta.exp().clamp(min=1e-9, max=1e8)
            d = _poisson_deviance(Y, mu)
            deviance.append(d)
            n_iter = it + 1

            if it > 0 and abs(prev_dev - d) / (abs(prev_dev) + 1e-9) < tol:
                break
            prev_dev = d

            # Z update
            resid = Y - mu
            grad_Z = resid.t() @ V - lam * Z
            H_Z = torch.einsum("mn,ml,mk->nlk", mu, V, V) + lam * eye_L
            try:
                Z = Z + torch.linalg.solve(H_Z, grad_Z.unsqueeze(-1)).squeeze(-1)
            except Exception:
                Z = Z + 0.01 * grad_Z

            # V update
            eta = a.unsqueeze(1) + V @ Z.t()
            mu = eta.exp().clamp(min=1e-9, max=1e8)
            resid = Y - mu
            grad_V = resid @ Z - lam * V
            H_V = torch.einsum("mn,nl,nk->mlk", mu, Z, Z) + lam * eye_L
            try:
                V = V + torch.linalg.solve(H_V, grad_V.unsqueeze(-1)).squeeze(-1)
            except Exception:
                V = V + 0.01 * grad_V

            # intercept update
            eta = a.unsqueeze(1) + V @ Z.t()
            mu = eta.exp().clamp(min=1e-9, max=1e8)
            ga = (Y - mu).sum(dim=1)
            ha = mu.sum(dim=1).clamp(min=1e-3)
            a = a + ga / ha

    return (
        Z.cpu().numpy().astype(np.float32),
        V.cpu().numpy().astype(np.float32),
        a.cpu().numpy().astype(np.float32),
        np.asarray(deviance, dtype=np.float32),
        n_iter,
    )
