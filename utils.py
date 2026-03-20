#!/usr/bin/env python3
import numpy as np
import torch
from typing import Literal, Tuple, Optional

from scipy.special import kv, gamma
from sklearn.gaussian_process.kernels import ConstantKernel as C, RBF, Matern

Metric = Literal["euclidean", "sqeuclidean", "manhattan", "cosine"]

def sample_uniform_disk(radius: float = 1.0, center=(0.0, 0.0), *, rng=None, gen: str = "numpy", num_type=torch.float64, device: torch.device | None = None):
    """
    Draw n points uniformly from a 2D disk.

    This is batching-consistent:
    sampling two batches in a row gives the same result as
    sampling one batch of the combined size, provided the same RNG
    is used and the outputs are concatenated.
    """
    if gen == "numpy":
    	if rng is None:
        	rng = np.random.default_rng()
    	def sampling_fn(*, size: (int, int)):
    		assert len(size)==2 and size[1]==2, "Disk sampling only in 2D!"
    		u = rng.uniform(0.0, 1.0, size=size)
    		u = torch.from_numpy(u).to(device=device, dtype=num_type, non_blocking=(device.type == "cuda"))
    		
    		theta = 2.0 * np.pi * u[:, 0]
    		r = radius * torch.sqrt(u[:, 1])
    		x = center[0] + r * torch.cos(theta)
    		y = center[1] + r * torch.sin(theta)
    		
    		return torch.column_stack((x, y))
    else:
    	if rng is None:
    		rng = torch.Generator(device=device)
    	def sampling_fn(*, size: (int, int)):
    		u = torch.rand(size=size, generator=rng, device=device, dtype=num_type)
    		
    		theta = 2.0 * np.pi * u[:, 0]
    		r = radius * torch.sqrt(u[:, 1])
    		x = center[0] + r * torch.cos(theta)
    		y = center[1] + r * torch.sin(theta)
    		
    		return torch.column_stack((x, y))

    return sampling_fn
    
def sample_gaussian(mean=0.0, std=1.0, *, rng=None, gen: str = "numpy", num_type=torch.float32, device: torch.device | None = None):
    """
    Draw n points uniformly from a 2D disk.

    This is batching-consistent:
    sampling two batches in a row gives the same result as
    sampling one batch of the combined size, provided the same RNG
    is used and the outputs are concatenated.
    """
    if gen == "numpy":
    	if rng is None:
        	rng = np.random.default_rng()
    	def sampling_fn(*, size: (int, int)):
        	sample = rng.normal(loc=mean, scale=std, size=size)
        	sample = torch.from_numpy(sample).to(device=device, dtype=num_type, non_blocking=(device.type == "cuda"))
        	return sample
    else:
    	if rng is None:
    		rng = torch.Generator(device=device)
    	def sampling_fn(*, size: (int, int)):
    		sample = torch.randn(size=size, generator=rng, device=device, dtype=num_type)
    		sample = sample * std
    		sample = sample + mean
    		return sample


    return sampling_fn
    
def dedupe(seq):
    """Remove duplicates while preserving original order."""
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out
    
def knn(
    x: np.ndarray,
    X_N: np.ndarray,
    k: int,
    *,
    metric: Metric = "euclidean",
    exclude_self: bool = False,
    self_index: Optional[int] = None,
    eps: float = 1e-12,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return indices and distances of the k nearest neighbors of x in X_N.

    Args:
        x: shape (d,) query point.
        X_N: shape (N, d) database points.
        k: number of neighbors requested.
        metric: "euclidean" | "sqeuclidean" | "manhattan" | "cosine".
        exclude_self: if True, exclude x if it appears in X_N.
        self_index: if you know x == X_N[self_index], pass it to exclude precisely.
        eps: small constant for cosine normalization stability.

    Returns:
        (idx, dist):
          idx: shape (k_eff,) int indices into X_N, sorted nearest->farthest
          dist: shape (k_eff,) distances (per chosen metric), sorted
        where k_eff = min(k, N - (1 if exclude_self else 0)) (and never negative).
    """
    x = np.asarray(x)
    X_N = np.asarray(X_N)

    if X_N.ndim != 2:
        raise ValueError(f"X_N must be 2D (N,d). Got shape {X_N.shape}.")
    if x.ndim != 1 or x.shape[0] != X_N.shape[1]:
        raise ValueError(f"x must be 1D with length d={X_N.shape[1]}. Got shape {x.shape}.")
    if k <= 0:
        raise ValueError("k must be a positive integer.")

    N = X_N.shape[0]
    if N == 0:
        return np.empty((0,), dtype=int), np.empty((0,), dtype=float)

    # Compute distances
    if metric in ("euclidean", "sqeuclidean"):
        diff = X_N - x  # (N,d)
        d2 = np.einsum("ij,ij->i", diff, diff)  # squared L2, shape (N,)
        dist = np.sqrt(d2) if metric == "euclidean" else d2
    elif metric == "manhattan":
        dist = np.sum(np.abs(X_N - x), axis=1)
    elif metric == "cosine":
        x_norm = np.linalg.norm(x) + eps
        X_norm = np.linalg.norm(X_N, axis=1) + eps
        # cosine distance = 1 - cosine similarity
        sim = (X_N @ x) / (X_norm * x_norm)
        dist = 1.0 - sim
    else:
        raise ValueError(f"Unknown metric: {metric}")

    # Optionally exclude self
    if exclude_self:
        if self_index is not None:
            if not (0 <= self_index < N):
                raise ValueError("self_index out of range.")
            dist = dist.copy()
            dist[self_index] = np.inf
        else:
            # Exclude any exact match row(s) if present
            matches = np.all(X_N == x, axis=1)
            if np.any(matches):
                dist = dist.copy()
                dist[matches] = np.inf

    # Effective k
    k_eff = min(k, N - (1 if (exclude_self and np.isfinite(dist).any()) else 0))
    k_eff = max(k_eff, 0)

    # If everything got excluded (e.g., N=1 and exclude_self=True)
    finite_mask = np.isfinite(dist)
    if k_eff == 0 or not np.any(finite_mask):
        return np.empty((0,), dtype=int), np.empty((0,), dtype=float)

    # Select k smallest efficiently, then sort them
    # (argpartition gives unsorted top-k; then we sort those)
    idx_part = np.argpartition(dist, kth=min(k_eff - 1, N - 1))[:k_eff]
    order = np.argsort(dist[idx_part])
    idx = idx_part[order]
    return idx.astype(int), dist[idx].astype(float)
    
# ============================================================
# KERNEL FACTORY
#   - returns both a scikit-learn kernel object and a lightweight spec
# ============================================================
def make_kernel(kind, sigma_f2, ell, *, nu=1.5):
    """
    Supported kinds:
        "RBF"
        "Matern"

    Returns
    -------
    sk_kernel : sklearn kernel object
    spec : dict
        Internal kernel description used by the chunked batched routines.
    """
    if sigma_f2 <= 0:
        raise ValueError("sigma_f2 must be positive")
    if ell <= 0:
        raise ValueError("ell must be positive")

    kind_l = kind.lower()

    if kind_l == "rbf":
        sk_kernel = C(
            constant_value=sigma_f2,
            constant_value_bounds="fixed",
        ) * RBF(
            length_scale=ell,
            length_scale_bounds="fixed",
        )
        spec = {
            "kind": "rbf",
            "sigma_f2": float(sigma_f2),
            "ell": float(ell),
        }

    elif kind_l == "matern":
        if nu <= 0:
            raise ValueError("For Matern, nu must be positive")

        sk_kernel = C(
            constant_value=sigma_f2,
            constant_value_bounds="fixed",
        ) * Matern(
            length_scale=ell,
            length_scale_bounds="fixed",
            nu=nu,
        )
        spec = {
            "kind": "matern",
            "sigma_f2": float(sigma_f2),
            "ell": float(ell),
            "nu": float(nu),
        }

    else:
        raise ValueError(f"Unsupported kernel kind: {kind}")

    return sk_kernel, spec
    
# ============================================================
# BATCHED DISTANCES WITHOUT FORMING A HUGE (batch,p,q,d) ARRAY
# ============================================================
def pairwise_sq_dists_batched(A, B):
    """
    A: shape (batch, p, d)
    B: shape (batch, q, d)

    Returns
    -------
    D2 : shape (batch, p, q)
        D2[b, i, j] = ||A[b, i, :] - B[b, j, :]||^2
    """
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)

    A2 = np.sum(A * A, axis=-1, keepdims=True)          # (batch, p, 1)
    B2 = np.sum(B * B, axis=-1)[:, None, :]            # (batch, 1, q)
    cross = np.einsum("bpd,bqd->bpq", A, B, optimize=True)

    D2 = A2 + B2 - 2.0 * cross
    return np.maximum(D2, 0.0)
    
# ============================================================
# STABLE Matern factor
# M_nu(z) = 2^(1-nu)/Gamma(nu) * z^nu * K_nu(z), with M_nu(0)=1
# ============================================================
def matern_factor(z, nu):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)

    zero = (z == 0.0)
    out[zero] = 1.0

    nonzero = ~zero

    # Common fast special cases
    if nu == 0.5:
        out[nonzero] = np.exp(-z[nonzero])
        return out

    if nu == 1.5:
        zz = z[nonzero]
        out[nonzero] = (1.0 + zz) * np.exp(-zz)
        return out

    if nu == 2.5:
        zz = z[nonzero]
        out[nonzero] = (1.0 + zz + zz**2 / 3.0) * np.exp(-zz)
        return out

    # Small-z branch for better stability when 0 < nu < 1
    if 0.0 < nu < 1.0:
        small = nonzero & (z < 1e-6)
        if np.any(small):
            c = gamma(-nu) / (2.0 ** (2.0 * nu) * gamma(nu))
            out[small] = 1.0 + c * z[small] ** (2.0 * nu)

        reg = nonzero & (~small)
    else:
        reg = nonzero

    if np.any(reg):
        zr = z[reg]
        out[reg] = (2.0 ** (1.0 - nu) / gamma(nu)) * (zr ** nu) * kv(nu, zr)

    return out
    
# ============================================================
# BATCHED KERNEL EVALUATION
# ============================================================
def kernel_batched(A, B, spec):
    """
    A: shape (batch, p, d)
    B: shape (batch, q, d)

    Returns
    -------
    K : shape (batch, p, q)
    """
    D2 = pairwise_sq_dists_batched(A, B)

    kind = spec["kind"]
    sigma_f2 = spec["sigma_f2"]
    ell = spec["ell"]

    if kind == "rbf":
        return sigma_f2 * np.exp(-0.5 * D2 / (ell ** 2))

    if kind == "matern":
        nu = spec["nu"]
        r = np.sqrt(D2)  # Matern uses ordinary distance, not squared distance
        z = np.sqrt(2.0 * nu) * r / ell
        return sigma_f2 * matern_factor(z, nu)

    raise RuntimeError("Unknown kernel kind")


def kernel_diag_batched(A, spec):
    """
    A: shape (batch, p, d)

    Returns
    -------
    diag_vals : shape (batch, p)
        k(x, x) for each point
    """
    return np.full(A.shape[:2], spec["sigma_f2"], dtype=np.float64)

# ============================================================
# BATCHED CHOLESKY SOLVE
# ============================================================
def cholesky_solve_batched(L, rhs):
    """
    Solve (L L^T) x = rhs for a stack of matrices.

    Parameters
    ----------
    L : ndarray, shape (batch, m, m)
        Lower-triangular Cholesky factors.

    rhs : ndarray, shape (batch, m, k)

    Returns
    -------
    sol : ndarray, shape (batch, m, k)
    """
    y = np.linalg.solve(L, rhs)
    x = np.linalg.solve(np.swapaxes(L, -1, -2), y)
    return x
    
def f_bounded_lipschitz(X):
    """
    Evaluate the bounded globally Lipschitz function

        f_d(x) = tanh(
            (1/sqrt(d)) * sum_{j=1}^d sin(sqrt(d) * x_j)
            + (1/sqrt(d/2)) * sum_{j=1}^{d/2} cos(sqrt(d) * (x_{2j-1} + x_{2j}))
        )

    on a batch X of shape (b, d).

    Parameters
    ----------
    X : ndarray, shape (b, d)

    Returns
    -------
    out : ndarray, shape (b,)
    """
    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2:
        raise ValueError("X must have shape (b, d)")

    b, d = X.shape
    if d % 2 != 0:
        raise ValueError("f_bounded_lipschitz expects even input dimension d")

    sqrt_d = np.sqrt(d)

    term1 = np.sum(np.sin(sqrt_d * X), axis=1) / sqrt_d

    x_odd = X[:, 0::2]
    x_even = X[:, 1::2]
    term2 = np.sum(np.cos(sqrt_d * (x_odd + x_even)), axis=1) / np.sqrt(d / 2.0)

    return np.tanh(term1 + term2)



