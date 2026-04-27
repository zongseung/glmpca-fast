//! GLM-PCA algorithm core (Townes, Hicks, Aryee, Irizarry 2019).
//!
//! Implements coordinate-block Newton optimisation of the Poisson GLM-PCA
//! deviance with L2 regularisation on factors and loadings.
//!
//! Model:
//!     Y_{ij} ~ Poisson(μ_{ij}),   μ_{ij} = exp(a_i + V_i· · Z_j·)
//! Loss:
//!     L(V, Z, a) = Σ_{ij} dev(Y_{ij}, μ_{ij}) + λ (‖V‖² + ‖Z‖²)
//!
//! Per outer iteration we update Z (per-sample 8×8 Newton), V
//! (per-feature 8×8 Newton), and intercepts a (scalar Newton). All
//! per-row updates parallelise with rayon.
//!
//! For genotype dosage data X ∈ {0,1,2} with MAF ≥ 0.01, Poisson is a
//! reasonable approximation. For richer dispersion modelling pass the
//! same data through the negative-binomial branch (TODO; not yet
//! implemented).

use ndarray::{s, Array1, Array2, ArrayView2, Axis};
use rayon::prelude::*;

/// Configuration for the GLM-PCA solver.
#[derive(Clone, Copy)]
pub struct GlmPcaConfig {
    pub l: usize,
    pub max_iter: usize,
    pub tol: f32,
    pub penalty: f32,
    pub seed: u64,
}

/// Output of a single GLM-PCA fit.
pub struct GlmPcaFit {
    pub factors: Array2<f32>,    // (N, L)
    pub loadings: Array2<f32>,   // (M, L)
    pub intercept: Array1<f32>,  // (M,)
    pub deviance: Vec<f32>,
    pub n_iter: usize,
}

/// Fit GLM-PCA (Poisson family) to a dosage matrix.
///
/// `y` must be (M=variants, N=samples); each entry should be a non-negative
/// count. For dosage data ∈ {0, 1, 2} this is satisfied directly.
pub fn fit_poisson(y: ArrayView2<f32>, cfg: GlmPcaConfig) -> GlmPcaFit {
    let (m, n) = y.dim();
    let l = cfg.l.min(m).min(n);
    let lam = cfg.penalty.max(1e-6);

    // Random init via deterministic LCG (no external rand crate).
    let mut rng_state = cfg.seed.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let mut next = || -> f32 {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (rng_state >> 32) as i32 as f32 / (i32::MAX as f32) * 0.05
    };

    let mut z = Array2::<f32>::from_shape_fn((n, l), |_| next());
    let mut v = Array2::<f32>::from_shape_fn((m, l), |_| next());
    // Intercept = log(mean(y_i) + 1) for stability
    let mut a = y
        .map_axis(Axis(1), |row| (row.mean().unwrap_or(0.0) + 1.0).ln())
        .into_owned();

    let mut deviance_history = Vec::with_capacity(cfg.max_iter);
    let mut prev_dev = f32::INFINITY;
    let mut iter_count = 0usize;

    for iter in 0..cfg.max_iter {
        // Compute μ = exp(a + V Z')   [shape (M, N)]
        let eta = compute_eta(&v.view(), &z.view(), &a.view());
        let mu = eta.mapv(f32::exp);

        // Deviance
        let dev = poisson_deviance(&y, &mu.view());
        deviance_history.push(dev);
        iter_count = iter + 1;

        if iter > 0 {
            let rel = (prev_dev - dev).abs() / (prev_dev.abs() + 1e-9);
            if rel < cfg.tol {
                break;
            }
        }
        prev_dev = dev;

        // ---- Update Z (per-sample Newton) ----
        let resid = &y - &mu;
        update_z(&mut z, &v.view(), &resid.view(), &mu.view(), lam, l);

        // ---- Update V (per-feature Newton) ----
        // Recompute μ since z changed
        let eta2 = compute_eta(&v.view(), &z.view(), &a.view());
        let mu2 = eta2.mapv(f32::exp);
        let resid2 = &y - &mu2;
        update_v(&mut v, &z.view(), &resid2.view(), &mu2.view(), lam, l);

        // ---- Update intercepts a ----
        let eta3 = compute_eta(&v.view(), &z.view(), &a.view());
        let mu3 = eta3.mapv(f32::exp);
        update_a(&mut a, &y, &mu3.view());
    }

    GlmPcaFit {
        factors: z,
        loadings: v,
        intercept: a,
        deviance: deviance_history,
        n_iter: iter_count,
    }
}

fn compute_eta(v: &ArrayView2<f32>, z: &ArrayView2<f32>, a: &ndarray::ArrayView1<f32>) -> Array2<f32> {
    // η = V · Z' + a (broadcast a to columns)
    let mut eta = v.dot(&z.t());
    for (i, mut row) in eta.axis_iter_mut(Axis(0)).enumerate() {
        let ai = a[i];
        row.iter_mut().for_each(|x| *x += ai);
    }
    eta
}

/// Poisson deviance: 2 Σ [y log(y/μ) − (y − μ)].
fn poisson_deviance(y: &ArrayView2<f32>, mu: &ArrayView2<f32>) -> f32 {
    let mut sum = 0.0f64;
    ndarray::Zip::from(y).and(mu).for_each(|&yi, &mi| {
        let mi = mi.max(1e-9);
        let term = if yi > 0.0 {
            (yi as f64) * ((yi as f64 / mi as f64).ln()) - (yi as f64 - mi as f64)
        } else {
            -(yi as f64 - mi as f64)
        };
        sum += 2.0 * term;
    });
    sum.max(0.0) as f32
}

/// Update Z: for each sample j, Newton step in 8×8 latent space.
fn update_z(
    z: &mut Array2<f32>,
    v: &ArrayView2<f32>,
    resid: &ArrayView2<f32>,
    mu: &ArrayView2<f32>,
    lam: f32,
    l: usize,
) {
    let n = z.shape()[0];
    let updates: Vec<(usize, Vec<f32>)> = (0..n)
        .into_par_iter()
        .map(|j| {
            // ∇L_z = V' (resid_j) - λ z_j
            let resid_j = resid.column(j);
            let mu_j = mu.column(j);
            let grad: Vec<f32> = (0..l)
                .map(|k| {
                    let mut s = 0.0f32;
                    for i in 0..v.shape()[0] {
                        s += v[[i, k]] * resid_j[i];
                    }
                    s - lam * z[[j, k]]
                })
                .collect();
            // Hessian H = V' diag(μ_j) V + λ I, shape (L, L)
            let mut h = vec![0.0f32; l * l];
            for k1 in 0..l {
                for k2 in k1..l {
                    let mut s = 0.0f32;
                    for i in 0..v.shape()[0] {
                        s += v[[i, k1]] * mu_j[i] * v[[i, k2]];
                    }
                    if k1 == k2 {
                        s += lam;
                    }
                    h[k1 * l + k2] = s;
                    h[k2 * l + k1] = s;
                }
            }
            // Solve H · δ = grad via small dense Gaussian elimination
            let delta = solve_small(&h, &grad, l);
            (j, delta)
        })
        .collect();

    for (j, delta) in updates {
        for k in 0..l {
            z[[j, k]] += delta[k];
        }
    }
}

/// Update V: per-feature i, Newton step.
fn update_v(
    v: &mut Array2<f32>,
    z: &ArrayView2<f32>,
    resid: &ArrayView2<f32>,
    mu: &ArrayView2<f32>,
    lam: f32,
    l: usize,
) {
    let m = v.shape()[0];
    let n = z.shape()[0];
    let updates: Vec<(usize, Vec<f32>)> = (0..m)
        .into_par_iter()
        .map(|i| {
            let resid_i = resid.row(i);
            let mu_i = mu.row(i);
            let grad: Vec<f32> = (0..l)
                .map(|k| {
                    let mut s = 0.0f32;
                    for j in 0..n {
                        s += z[[j, k]] * resid_i[j];
                    }
                    s - lam * v[[i, k]]
                })
                .collect();
            let mut h = vec![0.0f32; l * l];
            for k1 in 0..l {
                for k2 in k1..l {
                    let mut s = 0.0f32;
                    for j in 0..n {
                        s += z[[j, k1]] * mu_i[j] * z[[j, k2]];
                    }
                    if k1 == k2 {
                        s += lam;
                    }
                    h[k1 * l + k2] = s;
                    h[k2 * l + k1] = s;
                }
            }
            let delta = solve_small(&h, &grad, l);
            (i, delta)
        })
        .collect();

    for (i, delta) in updates {
        for k in 0..l {
            v[[i, k]] += delta[k];
        }
    }
}

fn update_a(a: &mut Array1<f32>, y: &ArrayView2<f32>, mu: &ArrayView2<f32>) {
    let m = a.len();
    for i in 0..m {
        let yi: f32 = y.row(i).sum();
        let mi: f32 = mu.row(i).sum();
        let denom = mi.max(1e-3);
        a[i] += (yi - mi) / denom;
    }
}

/// Solve a small (L ≤ ~32) symmetric positive-definite system H·x = b via
/// Gauss–Jordan elimination. We avoid a LAPACK dependency since GLM-PCA's
/// per-row systems are tiny.
fn solve_small(h: &[f32], b: &[f32], l: usize) -> Vec<f32> {
    // Augmented matrix [H | b], dimension l × (l+1)
    let mut a = vec![0.0f32; l * (l + 1)];
    for i in 0..l {
        for j in 0..l {
            a[i * (l + 1) + j] = h[i * l + j];
        }
        a[i * (l + 1) + l] = b[i];
    }
    // Forward elimination with partial pivoting
    for col in 0..l {
        // Find pivot
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
            // Singular — return zeros (no update). Caller treats as fallback.
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
