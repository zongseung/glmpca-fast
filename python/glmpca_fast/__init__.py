"""glmpca-fast — fast GLM-PCA (Townes et al. 2019) with Rust + optional GPU.

Public API
----------
    fit_poisson(Y, L, ...)        Single-gene Poisson GLM-PCA fit.
    project_ols(X_held, ...)      OLS projection of held-out samples.

Backends
--------
* ``backend="rust"`` (default) — Rust + rayon, ~13× faster than glmpca-py.
* ``backend="torch"``           — PyTorch CUDA, ~290× faster on GPU.
                                  Requires ``pip install glmpca-fast[torch]``.
* ``backend="auto"``             — picks torch when CUDA is available.

Reference
---------
Townes, F. W., Hicks, S. C., Aryee, M. J., & Irizarry, R. A. (2019).
"Feature selection and dimension reduction for single-cell RNA-Seq based
on a multinomial model." *Genome Biology*, 20:295.
doi:10.1186/s13059-019-1861-6
"""

from __future__ import annotations

from typing import Literal

import numpy as np

from glmpca_fast import _rust  # type: ignore

__all__ = ["fit_poisson", "project_ols", "__version__"]
__version__ = "0.1.0"

Backend = Literal["rust", "torch", "auto"]


def _resolve_backend(backend: Backend) -> str:
    if backend not in ("rust", "torch", "auto"):
        raise ValueError(f"Unknown backend {backend!r}; expected rust|torch|auto")
    if backend == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "torch"
        except ImportError:
            pass
        return "rust"
    if backend == "torch":
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "torch backend requires PyTorch — install via "
                "`pip install glmpca-fast[torch]`"
            ) from exc
    return backend


def fit_poisson(
    Y: np.ndarray,
    L: int,
    *,
    max_iter: int = 100,
    tol: float = 1e-4,
    penalty: float = 1.0,
    seed: int = 42,
    backend: Backend = "rust",
    device: str | None = None,
) -> dict:
    """Fit Poisson GLM-PCA to a count / dosage matrix.

    Parameters
    ----------
    Y : ndarray, shape (n_samples, n_features)
        Non-negative observations. For genotype dosage data this is in
        ``{0, 1, 2}``.
    L : int
        Latent dimensionality.
    max_iter : int
        Outer-iteration cap (default 100).
    tol : float
        Relative-deviance convergence threshold (default 1e-4).
    penalty : float
        L2 ridge on factors and loadings (default 1.0).
    seed : int
        Random init seed (default 42).
    backend : {'rust', 'torch', 'auto'}
        Compute backend. ``'auto'`` picks ``'torch'`` when a CUDA device is
        present, otherwise ``'rust'``.
    device : str | None
        Torch device override (e.g. ``'cuda:1'``). Only honoured for the
        torch backend; ignored otherwise.

    Returns
    -------
    dict with keys
        ``factors``  (n_samples, L)
        ``loadings`` (n_features, L)
        ``intercept`` (n_features,)
        ``deviance`` length-(n_iter+1) trajectory
        ``n_iter``   actual iterations taken
        ``backend``  which backend was used
    """
    Y = np.ascontiguousarray(np.asarray(Y, dtype=np.float32))
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2-D, got shape {Y.shape}")
    if L < 2:
        raise ValueError(f"L must be >= 2, got {L}")
    if Y.shape[0] < L or Y.shape[1] < L:
        raise ValueError(f"Need n_samples and n_features >= L; got {Y.shape}, L={L}")

    chosen = _resolve_backend(backend)

    if chosen == "rust":
        f, v, intercept, dev, n_iter = _rust.fit_poisson(
            Y, l=L, max_iter=max_iter, tol=tol, penalty=penalty, seed=seed,
        )
        return {
            "factors": np.asarray(f, dtype=np.float32),
            "loadings": np.asarray(v, dtype=np.float32),
            "intercept": np.asarray(intercept, dtype=np.float32),
            "deviance": np.asarray(dev, dtype=np.float32),
            "n_iter": int(n_iter),
            "backend": "rust",
        }

    # torch backend
    from glmpca_fast._torch import fit_poisson_torch

    f, v, intercept, dev, n_iter = fit_poisson_torch(
        Y, L, max_iter=max_iter, tol=tol, penalty=penalty,
        seed=seed, device=device,
    )
    return {
        "factors": f,
        "loadings": v,
        "intercept": intercept,
        "deviance": dev,
        "n_iter": n_iter,
        "backend": "torch",
    }


def project_ols(
    X_held: np.ndarray,
    train_mean: np.ndarray,
    loadings: np.ndarray,
) -> np.ndarray:
    """Project held-out samples onto a fitted GLM-PCA basis via OLS.

    Approximate-but-fast: uses the Pearson-residual approximation and
    solves :math:`(V^\\top V) Z = V^\\top X_{\\text{held}}` per row.

    Parameters
    ----------
    X_held : ndarray, shape (n_held, n_features)
    train_mean : ndarray, shape (n_features,)
        Feature-wise mean of the training rows used to fit GLM-PCA.
    loadings : ndarray, shape (n_features, L)

    Returns
    -------
    ndarray, shape (n_held, L)
    """
    X_held = np.ascontiguousarray(X_held, dtype=np.float32)
    train_mean = np.ascontiguousarray(train_mean, dtype=np.float32)
    loadings = np.ascontiguousarray(loadings, dtype=np.float32)
    return np.asarray(_rust.project_ols(X_held, train_mean, loadings), dtype=np.float32)
