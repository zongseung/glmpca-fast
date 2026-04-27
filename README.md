# glmpca-fast

[![PyPI](https://img.shields.io/pypi/v/glmpca-fast.svg)](https://pypi.org/project/glmpca-fast/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Fast **GLM-PCA** (Generalised Linear Model PCA) for non-Gaussian count
data — Rust core with optional **PyTorch GPU** backend.

Implements the algorithm of:

> Townes, F. W., Hicks, S. C., Aryee, M. J., & Irizarry, R. A. (2019).
> "Feature selection and dimension reduction for single-cell RNA-Seq based
> on a multinomial model." *Genome Biology*, 20:295.
> [doi:10.1186/s13059-019-1861-6](https://doi.org/10.1186/s13059-019-1861-6)

## Why

Standard PCA assumes Gaussian + homoscedastic noise — wrong likelihood for
count data (RNA-seq UMIs, genotype dosages, etc.). GLM-PCA fits the
proper Poisson / Multinomial / Bernoulli / NB likelihood, capturing the
mean–variance relationship inside the model.

The reference implementation
([willtownes/glmpca-py](https://github.com/willtownes/glmpca-py)) is
CPU-only Python. **glmpca-fast** ports the algorithm to:

* **Rust** (rayon-parallel coordinate-block Newton) — ~13× faster than
  glmpca-py on a single CPU.
* **PyTorch** with batched ``torch.linalg.solve`` — ~290× faster on a
  modern GPU (RTX A6000) per fit.

## Install

### With [uv](https://docs.astral.sh/uv/) (recommended)

```bash
# Add to a project (preferred)
uv add glmpca-fast                    # CPU (Rust + numpy)
uv add "glmpca-fast[torch]"           # + GPU (PyTorch)

# Or install into the active environment
uv pip install glmpca-fast
uv pip install "glmpca-fast[torch]"

# Or run a one-shot script
uv tool run --from glmpca-fast python -c "from glmpca_fast import fit_poisson; ..."
```

### With pip

```bash
pip install glmpca-fast              # CPU
pip install "glmpca-fast[torch]"     # + GPU
```

## Quick start

```python
import numpy as np
from glmpca_fast import fit_poisson

# Synthetic Binomial(2, p) genotype dosage matrix
rng = np.random.default_rng(0)
N, M, L = 2504, 200, 8
p = rng.uniform(0.05, 0.5, M)
Y = rng.binomial(2, p, size=(N, M)).astype(np.float32)

# Rust backend (default, CPU)
res = fit_poisson(Y, L=L, max_iter=100)
print(res["factors"].shape)         # (2504, 8)
print(res["loadings"].shape)        # (200, 8)
print(res["deviance"][-1])          # final deviance
print(res["backend"])               # 'rust'

# Auto backend — picks GPU if CUDA is available
res = fit_poisson(Y, L=L, backend="auto")
print(res["backend"])               # 'torch' if CUDA, else 'rust'

# Explicit GPU device
res = fit_poisson(Y, L=L, backend="torch", device="cuda:0")
```

## API

```python
fit_poisson(
    Y,                   # (n_samples, n_features) non-negative counts
    L,                   # latent dim, >= 2
    max_iter=100,
    tol=1e-4,            # relative deviance tolerance
    penalty=1.0,         # L2 ridge on factors and loadings
    seed=42,
    backend="rust",      # 'rust' | 'torch' | 'auto'
    device=None,         # torch device override (e.g. 'cuda:1')
) -> dict
# returns: factors, loadings, intercept, deviance, n_iter, backend

project_ols(X_held, train_mean, loadings) -> ndarray
# Approximate OLS projection of held-out samples (Pearson-residual approx).
```

## Benchmark

Single gene, 2,504 samples × 200 variants, L=8:

| Backend | Time / fit | Speedup vs `glmpca-py` |
| --- | --- | --- |
| `glmpca-py` (reference, NumPy) | 7.58 s | 1× |
| **glmpca-fast (Rust, 11 cores)** | **0.56 s** | **13.5×** |
| **glmpca-fast (PyTorch, RTX A6000)** | **0.026 s** | **290×** |

Both backends converge to within ~1 % of the reference final deviance
(non-convex objective, different random init).

## Limitations / scope

* Currently **Poisson family only**. Multinomial / NB / Bernoulli
  branches are planned for v0.2.
* Newton step uses full update without line-search damping. For
  degenerate Hessians the implementation falls back to a small
  gradient step.
* Held-out projection uses the OLS approximation (not full per-sample
  IRLS).
* Built and tested on Linux x86-64 + CUDA 12. Other platforms via
  source build.

## Citation

If you use this package, please cite **both** the original paper and
this software:

```bibtex
@article{Townes2019GLMPCA,
  title   = {Feature selection and dimension reduction for single-cell
             RNA-Seq based on a multinomial model},
  author  = {Townes, F. William and Hicks, Stephanie C. and Aryee,
             Martin J. and Irizarry, Rafael A.},
  journal = {Genome Biology},
  volume  = {20},
  number  = {1},
  pages   = {295},
  year    = {2019},
  doi     = {10.1186/s13059-019-1861-6}
}

@software{glmpca_fast,
  title  = {glmpca-fast: Fast GLM-PCA with Rust and GPU backends},
  author = {zongseung},
  year   = {2026},
  url    = {https://github.com/zongseung/glmpca-fast}
}
```

## License

MIT — see [`LICENSE`](LICENSE).
