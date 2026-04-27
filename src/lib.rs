//! Python bindings for the Rust GLM-PCA implementation.
//!
//! Mirrors the API contract of :mod:`src.preprocessing.glm_pca` (Python
//! fallback) so that the Python wrapper can transparently choose the
//! Rust implementation when available.
//!
//! See `glmpca.rs` for the algorithm and Townes et al. (2019) for the
//! original derivation.

use ndarray::s;
use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
use pyo3::prelude::*;

mod glmpca;

/// Fit GLM-PCA (Poisson family) to a dosage matrix and return factors,
/// loadings, intercept, deviance trajectory.
///
/// Args:
///     y: (n_samples, n_variants) float32 ndarray, dosage in [0, 2].
///     l: latent dimensionality.
///     max_iter: outer-iteration cap (default 100).
///     tol: relative-deviance convergence tolerance (default 1e-4).
///     penalty: L2 ridge on factors and loadings (default 1.0).
///     seed: deterministic init seed (default 42).
///
/// Returns:
///     (factors[n_samples, l], loadings[n_variants, l],
///      intercept[n_variants], deviance[n_iter+1], n_iter)
#[pyfunction]
#[pyo3(signature = (y, l, max_iter=100, tol=1e-4, penalty=1.0, seed=42))]
fn fit_poisson<'py>(
    py: Python<'py>,
    y: PyReadonlyArray2<'py, f32>,
    l: usize,
    max_iter: usize,
    tol: f32,
    penalty: f32,
    seed: u64,
) -> PyResult<(
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray2<f32>>,
    Bound<'py, PyArray1<f32>>,
    Vec<f32>,
    usize,
)> {
    // Input is (n_samples, n_variants); GLM-PCA expects (n_variants, n_samples).
    let y_view = y.as_array();
    let y_t = y_view.t().to_owned();

    let cfg = glmpca::GlmPcaConfig {
        l,
        max_iter,
        tol,
        penalty,
        seed,
    };
    let fit = py.allow_threads(|| glmpca::fit_poisson(y_t.view(), cfg));

    Ok((
        fit.factors.into_pyarray_bound(py),
        fit.loadings.into_pyarray_bound(py),
        fit.intercept.into_pyarray_bound(py),
        fit.deviance,
        fit.n_iter,
    ))
}

/// Project held-out samples onto a fitted GLM-PCA basis via OLS in the
/// Pearson-residual approximation. Matches the Python `_project_held_out`
/// helper so that train-only fit + test projection produces identical
/// results across the two backends.
///
/// Args:
///     x_held: (n_held, n_variants) float32 ndarray.
///     train_mean: (n_variants,) feature-wise mean from training rows.
///     loadings: (n_variants, l) GLM-PCA loadings from training fit.
///
/// Returns:
///     (n_held, l) projected factors.
#[pyfunction]
fn project_ols<'py>(
    py: Python<'py>,
    x_held: PyReadonlyArray2<'py, f32>,
    train_mean: PyReadonlyArray1<'py, f32>,
    loadings: PyReadonlyArray2<'py, f32>,
) -> PyResult<Bound<'py, PyArray2<f32>>> {
    let x = x_held.as_array();
    let mu = train_mean.as_array();
    let v = loadings.as_array();
    let (n_held, m) = x.dim();
    let l = v.shape()[1];

    // Centered X = (x - mu_train), shape (n_held, m)
    let mut centered = ndarray::Array2::<f32>::zeros((n_held, m));
    for i in 0..n_held {
        for j in 0..m {
            centered[[i, j]] = x[[i, j]] - mu[j];
        }
    }

    // Compute V'V (l × l)
    let vtv = v.t().dot(&v);

    // Pseudo-inverse via the small Gaussian-elimination solver in glmpca.rs.
    // We solve (V'V) · A = V'X' for A, but easier: project each row.
    // Z_held = centered · V · (V'V)^-1
    // First compute centered · V, shape (n_held, l)
    let cv = centered.dot(&v);

    // Solve (V'V + tiny ridge) · z' = cv' per row
    let mut h = vec![0.0f32; l * l];
    for k1 in 0..l {
        for k2 in 0..l {
            h[k1 * l + k2] = vtv[[k1, k2]];
            if k1 == k2 {
                h[k1 * l + k2] += 1e-6;
            }
        }
    }

    let mut out = ndarray::Array2::<f32>::zeros((n_held, l));
    for i in 0..n_held {
        let rhs: Vec<f32> = (0..l).map(|k| cv[[i, k]]).collect();
        let z = solve_small_inline(&h, &rhs, l);
        for k in 0..l {
            out[[i, k]] = z[k];
        }
    }
    Ok(out.into_pyarray_bound(py))
}

fn solve_small_inline(h: &[f32], b: &[f32], l: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; l * (l + 1)];
    for i in 0..l {
        for j in 0..l {
            a[i * (l + 1) + j] = h[i * l + j];
        }
        a[i * (l + 1) + l] = b[i];
    }
    for col in 0..l {
        let mut piv_row = col;
        let mut piv_val = a[col * (l + 1) + col].abs();
        for r in (col + 1)..l {
            let v = a[r * (l + 1) + col].abs();
            if v > piv_val {
                piv_val = v;
                piv_row = r;
            }
        }
        if piv_val < 1e-9 {
            return vec![0.0; l];
        }
        if piv_row != col {
            for c in 0..(l + 1) {
                a.swap(col * (l + 1) + c, piv_row * (l + 1) + c);
            }
        }
        let pv = a[col * (l + 1) + col];
        for c in col..(l + 1) {
            a[col * (l + 1) + c] /= pv;
        }
        for r in 0..l {
            if r == col {
                continue;
            }
            let factor = a[r * (l + 1) + col];
            for c in col..(l + 1) {
                a[r * (l + 1) + c] -= factor * a[col * (l + 1) + c];
            }
        }
    }
    (0..l).map(|i| a[i * (l + 1) + l]).collect()
}

#[pymodule]
fn _rust(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(fit_poisson, m)?)?;
    m.add_function(wrap_pyfunction!(project_ols, m)?)?;
    Ok(())
}
