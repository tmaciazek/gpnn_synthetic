import numpy as np
import sys
import os
import argparse

from utils import *

# ============================================================
# CHUNKED BATCHED GENERATION OF NNGP TRAINING RESPONSES
# ============================================================
def simulate_local_nngp_responses(
    X_test,
    X_train_local,
    *,
    k_gen_spec,
    sigma_xi2,
    chunk_size=512,
    rng=None,
    jitter=1e-10,
):
    """
    For each test point i, sample jointly
        [y(X_test[i]), y_train[i, 0], ..., y_train[i, m-1]]
    from the Gaussian law

        [y_* ; y] ~ N(mu, [[k(x_*,x_*) + sigma_xi2,   k(x_*, X)^T],
                           [k(X, x_*),               K(X,X) + sigma_xi2 I]])

    where
        y(x) = ||x||_2^2 + w(x) + eps,
        w ~ GP(0, k),
        eps ~ N(0, sigma_xi2).

    This is chunked over test points to control RAM usage.

    Parameters
    ----------
    X_test : ndarray, shape (n_test, d)
    X_train_local : ndarray, shape (n_test, m, d)
    k_gen_spec : dict
        Kernel specification created by make_kernel(...)[1].
    sigma_xi2 : float
        Observation-noise variance.
    chunk_size : int
        Number of test points processed at once.
    rng : np.random.Generator or None
    jitter : float
        Small diagonal term added for numerical stability.

    Returns
    -------
    y_test_true : ndarray, shape (n_test,)
        Response values at the test points.
    y_train : ndarray, shape (n_test, m)
        Response values at the local training points.
    """
    if rng is None:
        rng = np.random.default_rng()

    X_test = np.asarray(X_test, dtype=np.float64)
    X_train_local = np.asarray(X_train_local, dtype=np.float64)

    n_test, d = X_test.shape
    n_test2, m, d2 = X_train_local.shape

    if n_test2 != n_test or d2 != d:
        raise ValueError("Shape mismatch between X_test and X_train_local")
    if sigma_xi2 < 0:
        raise ValueError("sigma_xi2 must be nonnegative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    f_test_true = np.empty(n_test, dtype=np.float64)
    y_train = np.empty((n_test, m), dtype=np.float64)

    eye_m = np.eye(m, dtype=np.float64)
    eye_joint = np.eye(m + 1, dtype=np.float64)

    for start in range(0, n_test, chunk_size):
        stop = min(start + chunk_size, n_test)
        b = stop - start

        X_test_chunk = X_test[start:stop]         # (b, d)
        X_train_chunk = X_train_local[start:stop] # (b, m, d)
        X_star = X_test_chunk[:, None, :]         # (b, 1, d)

        # Mean function mu(x) = ||x||_2^2
        mu_star = np.sum(X_test_chunk**2, axis=1)       # (b,)
        mu_train = np.sum(X_train_chunk**2, axis=2)     # (b, m)

        # Covariance blocks for w
        c_star = kernel_diag_batched(X_star, k_gen_spec)[:, 0]             # (b,)
        k_star = kernel_batched(X_train_chunk, X_star, k_gen_spec)         # (b, m, 1)
        K_xx = kernel_batched(X_train_chunk, X_train_chunk, k_gen_spec)    # (b, m, m)

        # Observation covariance for y_train
        K_yy = K_xx + sigma_xi2 * eye_m[None, :, :]                        # (b, m, m)

        # Assemble the joint covariance of [y_* ; y_train]
        K_joint = np.empty((b, m + 1, m + 1), dtype=np.float64)
        K_joint[:, 0, 0] = c_star
        K_joint[:, 0, 1:] = k_star[:, :, 0]
        K_joint[:, 1:, 0] = k_star[:, :, 0]
        K_joint[:, 1:, 1:] = K_yy

        # Joint mean
        mu_joint = np.empty((b, m + 1), dtype=np.float64)
        mu_joint[:, 0] = mu_star
        mu_joint[:, 1:] = mu_train

        # Small numerical stabilizer
        K_joint += jitter * eye_joint[None, :, :]

        # Sample jointly
        L = np.linalg.cholesky(K_joint)                  # (b, m+1, m+1)
        z = rng.standard_normal(size=(b, m + 1, 1))
        joint_sample = mu_joint + (L @ z).squeeze(-1)   # (b, m+1)

        f_test_true[start:stop] = joint_sample[:, 0]
        y_train[start:stop] = joint_sample[:, 1:]

    return f_test_true, y_train

# ============================================================
# CHUNKED BATCHED NNGP PREDICTION WITH FIXED HYPERPARAMETERS
# ============================================================
def predict_local_nngp(
    X_test,
    X_train_local,
    y_train,
    *,
    k_pred_spec,
    sigma_xi2_hat,
    b1_hat,
    b2_hat,
    return_var=False,
    chunk_size=512,
    jitter=1e-10,
):
    """
    Computes the local NNGP posterior mean at each test point in chunks:
        mu_tilde_NNGP(x_*) = t_*^T b_hat + k_*^T K_N^{-1}(y_N - T_N b_hat)

    with
        f_b(x) = b1_hat * x1^2 + b2_hat * x2^2.

    No hyperparameter tuning is performed here:
    prediction is computed directly from the fixed kernel parameters
    in k_pred_spec, the fixed noise variance sigma_xi2_hat,
    and the fixed mean parameters b1_hat, b2_hat.

    Returns
    -------
    mu_pred : (n_test,)
    var_pred : (n_test,)
    """
    X_test = np.asarray(X_test, dtype=np.float64)
    X_train_local = np.asarray(X_train_local, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)

    n_test, d = X_test.shape
    n_test2, m, d2 = X_train_local.shape

    if n_test2 != n_test or d2 != d:
        raise ValueError("Shape mismatch between X_test and X_train_local")
    if y_train.shape != (n_test, m):
        raise ValueError("y_train must have shape (n_test, m)")
    if d < 2:
        raise ValueError("This mean function requires at least 2 input dimensions")
    if sigma_xi2_hat < 0:
        raise ValueError("sigma_xi2_hat must be nonnegative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    mu_pred = np.empty(n_test, dtype=np.float64)
    var_pred = np.empty(n_test, dtype=np.float64)

    eye_m = np.eye(m, dtype=np.float64)

    for start in range(0, n_test, chunk_size):
        stop = min(start + chunk_size, n_test)

        X_test_chunk = X_test[start:stop]                 # (b, d)
        X_train_chunk = X_train_local[start:stop]         # (b, m, d)
        y_chunk = y_train[start:stop]                     # (b, m)
        b = stop - start

        X_star = X_test_chunk[:, None, :]                 # (b, 1, d)

        # Mean function f_b(x) = b1_hat * x1^2 + b2_hat * x2^2
        mean_star = b1_hat * X_test_chunk[:, 0]**2 + b2_hat * X_test_chunk[:, 1]**2     # (b,)
        mean_train = b1_hat * X_train_chunk[:, :, 0]**2 + b2_hat * X_train_chunk[:, :, 1]**2  # (b, m)

        K = kernel_batched(X_train_chunk, X_train_chunk, k_pred_spec)
        K = K + (sigma_xi2_hat + jitter) * eye_m[None, :, :]  # (b, m, m)

        k_star = kernel_batched(X_train_chunk, X_star, k_pred_spec)  # (b, m, 1)
        c_star = kernel_diag_batched(X_star, k_pred_spec)[:, 0]      # (b,)

        L = np.linalg.cholesky(K)

        # Center y_train by T_N b_hat
        y_centered = y_chunk - mean_train
        alpha = cholesky_solve_batched(L, y_centered[..., None])     # (b, m, 1)

        # Add back t_*^T b_hat
        mu_chunk = mean_star + np.sum(k_star[:, :, 0] * alpha[:, :, 0], axis=1)

        if return_var:
            v = cholesky_solve_batched(L, k_star)                    # (b, m, 1)
            var_chunk = c_star - np.sum(k_star[:, :, 0] * v[:, :, 0], axis=1)
            var_chunk = np.maximum(var_chunk, 0.0)
            var_pred[start:stop] = var_chunk

        mu_pred[start:stop] = mu_chunk

    return mu_pred, var_pred
    
# ============================================================
# CHUNKED BATCHED GENERATION OF GPnn TRAINING RESPONSES
# ============================================================
def simulate_local_gpnn_responses(
    X_test,
    X_train_local,
    *,
    sigma_xi2,
    chunk_size=512,
    rng=None,
):
    """
    For each test point i, compute
        f(X_test[i])
    and local noisy training responses
        y_train[i, j] = f(X_train_local[i, j]) + eps_{i,j},
    where eps_{i,j} are i.i.d. N(0, sigma_xi2).

    This is chunked over test points to control RAM usage.

    Parameters
    ----------
    X_test : ndarray, shape (n_test, d)
    X_train_local : ndarray, shape (n_test, m, d)
    k_gen_spec : dict
        Unused; kept for interface compatibility.
    sigma_xi2 : float
        Observation-noise variance.
    chunk_size : int
        Number of test points processed at once.
    rng : np.random.Generator or None
    jitter : float
        Unused; kept for interface compatibility.

    Returns
    -------
    f_test_true : ndarray, shape (n_test,)
        Deterministic function values at the test points.
    y_train : ndarray, shape (n_test, m)
        Noisy training responses.
    """
    if rng is None:
        rng = np.random.default_rng()

    X_test = np.asarray(X_test, dtype=np.float64)
    X_train_local = np.asarray(X_train_local, dtype=np.float64)

    n_test, d = X_test.shape
    n_test2, m, d2 = X_train_local.shape

    if n_test2 != n_test or d2 != d:
        raise ValueError("Shape mismatch between X_test and X_train_local")
    if sigma_xi2 < 0:
        raise ValueError("sigma_xi2 must be nonnegative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    f_test_true = np.empty(n_test, dtype=np.float64)
    y_train = np.empty((n_test, m), dtype=np.float64)

    noise_std = np.sqrt(sigma_xi2)

    for start in range(0, n_test, chunk_size):
        stop = min(start + chunk_size, n_test)
        b = stop - start

        X_test_chunk = X_test[start:stop]         # (b, d)
        X_train_chunk = X_train_local[start:stop] # (b, m, d)

        # Deterministic test values
        f_test_chunk = f_bounded_lipschitz(X_test_chunk)   # (b,)

        # Deterministic local train values
        X_train_flat = X_train_chunk.reshape(b * m, d)     # (b*m, d)
        f_train_flat = f_bounded_lipschitz(X_train_flat)   # (b*m,)
        f_train_chunk = f_train_flat.reshape(b, m)         # (b, m)

        # Add i.i.d. Gaussian noise only to training responses
        eps_chunk = noise_std * rng.standard_normal(size=(b, m))
        y_train_chunk = f_train_chunk + eps_chunk

        f_test_true[start:stop] = f_test_chunk
        y_train[start:stop] = y_train_chunk

    return f_test_true, y_train
    
# ============================================================
# CHUNKED BATCHED GPnn PREDICTION WITH FIXED HYPERPARAMETERS
# ============================================================
def predict_local_gpnn(
    X_test,
    X_train_local,
    y_train,
    *,
    k_pred_spec,
    sigma_xi2_hat,
    return_var = False,
    chunk_size=512,
    jitter=1e-10,
):
    """
    Computes the local GP posterior at each test point in chunks.

    No hyperparameter tuning is performed at all:
    the prediction is computed directly from the fixed kernel parameters
    in k_pred_spec and the fixed noise variance sigma_xi2_hat.

    Returns
    -------
    mu_pred : (n_test,)
    var_pred : (n_test,)
    """
    X_test = np.asarray(X_test, dtype=np.float64)
    X_train_local = np.asarray(X_train_local, dtype=np.float64)
    y_train = np.asarray(y_train, dtype=np.float64)

    n_test, d = X_test.shape
    n_test2, m, d2 = X_train_local.shape

    if n_test2 != n_test or d2 != d:
        raise ValueError("Shape mismatch between X_test and X_train_local")
    if y_train.shape != (n_test, m):
        raise ValueError("y_train must have shape (n_test, m)")
    if sigma_xi2_hat < 0:
        raise ValueError("sigma_xi2_hat must be nonnegative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    mu_pred = np.empty(n_test, dtype=np.float64)
    var_pred = np.empty(n_test, dtype=np.float64)

    eye_m = np.eye(m, dtype=np.float64)

    for start in range(0, n_test, chunk_size):
        stop = min(start + chunk_size, n_test)

        X_test_chunk = X_test[start:stop]                 # (b, d)
        X_train_chunk = X_train_local[start:stop]         # (b, m, d)
        y_chunk = y_train[start:stop]                     # (b, m)
        b = stop - start

        X_star = X_test_chunk[:, None, :]                 # (b, 1, d)

        K = kernel_batched(X_train_chunk, X_train_chunk, k_pred_spec)
        K = K + (sigma_xi2_hat + jitter) * eye_m[None, :, :]  # (b, m, m)

        k_star = kernel_batched(X_train_chunk, X_star, k_pred_spec)  # (b, m, 1)
        c_star = kernel_diag_batched(X_star, k_pred_spec)[:, 0]      # (b,)

        L = np.linalg.cholesky(K)

        alpha = cholesky_solve_batched(L, y_chunk[..., None])         # (b, m, 1)
        mu_chunk = np.sum(k_star[:, :, 0] * alpha[:, :, 0], axis=1)
        
        if return_var:
        	v = cholesky_solve_batched(L, k_star)                         # (b, m, 1)
        	var_chunk = c_star - np.sum(k_star[:, :, 0] * v[:, :, 0], axis=1)
        	var_chunk = np.maximum(var_chunk, 0.0)
        	var_pred[start:stop] = var_chunk

        mu_pred[start:stop] = mu_chunk

    return mu_pred, var_pred
    
# FULL EXPERIMENT PIPELINE
# ============================================================
def run_experiment(
    mode, 
    X_test, 
    X_train_local, 
    nu=0.5, 
    sigma_f2=1., 
    ell=1., 
    sigma_xi2=0.1, 
    sigma_f2_hat=1., 
    ell_hat=1., 
    sigma_xi2_hat=0.1, 
    b1_hat=1., 
    b2_hat=1., 
    return_var=False, 
    rng_seed=None, 
    chunk_size=512
    ):
    rng = np.random.default_rng(rng_seed)

    # --------------------------------------------------------
    # Prediction kernel k
    # --------------------------------------------------------
    k_pred, k_pred_spec = make_kernel(
        kind="Matern",     
        sigma_f2=sigma_f2_hat,
        ell=ell_hat,
        nu=nu,
    )
    
    # --------------------------------------------------------
    # Generative kernel k_gen
    # --------------------------------------------------------
    
    if mode == "NNGP":
    	k_gen, k_gen_spec = make_kernel(
        	kind="Matern",   # or "RBF"
        	sigma_f2=sigma_f2,
        	ell=ell,
        	nu=nu,
        )

        # --------------------------------------------------------
        # Generate latent NNGP values and noisy local responses
        # --------------------------------------------------------
    	f_test_true, y_train = simulate_local_nngp_responses(
            X_test,
            X_train_local,
            k_gen_spec=k_gen_spec,
            sigma_xi2=sigma_xi2,
            chunk_size=chunk_size,
            rng=rng,
    	)


        # --------------------------------------------------------
        # Predict with fixed hyperparameters
        # --------------------------------------------------------
    	mu_pred, var_pred = predict_local_nngp(
            X_test,
            X_train_local,
            y_train,
            k_pred_spec=k_pred_spec,
            sigma_xi2_hat=sigma_xi2_hat,
            b1_hat = b1_hat,
            b2_hat = b2_hat,
            chunk_size=chunk_size,
            return_var=return_var,
    	)
    else:
        # --------------------------------------------------------
        # Generate latent GPnn values and noisy local responses
        # --------------------------------------------------------
        f_test_true, y_train = simulate_local_gpnn_responses(
            X_test,
            X_train_local,
            sigma_xi2=sigma_xi2,
            chunk_size=chunk_size,
            rng=rng,
    	)


        # --------------------------------------------------------
        # Predict with fixed hyperparameters
        # --------------------------------------------------------
        mu_pred, var_pred = predict_local_gpnn(
            X_test,
            X_train_local,
            y_train,
            k_pred_spec=k_pred_spec,
            sigma_xi2_hat=sigma_xi2_hat,
            chunk_size=chunk_size,
            return_var=return_var,
        )

    mse_vs_latent = np.mean((mu_pred - f_test_true) ** 2)

    return {
        "X_test": X_test,
        "X_train_local": X_train_local,
        "f_test_true": f_test_true,
        "y_train": y_train,
        "mu_pred": mu_pred,
        "var_pred": var_pred,
        "mse_vs_latent": mse_vs_latent,
        "k_pred": k_pred,
        "k_pred_spec": k_pred_spec,
    }


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--mode", type=str, default='GPnn', help="Regression mode: GPnn|NNGP")
	ap.add_argument("--distro", type=str, default='gaussian', help="Data distribution: gaussian|uniform_disk")
	ap.add_argument("--dim", type=int, required=True)
	ap.add_argument("--seed_train", type=int, default=0)
	ap.add_argument("--nu", type=float, default=0.5)
	ap.add_argument("--ell", type=float, default=1.)
	ap.add_argument("--sf2", type=float, default=1.)
	ap.add_argument("--sxi2", type=float, default=0.1)
	ap.add_argument("--b1_hat", type=float, default=0.5)
	ap.add_argument("--b2_hat", type=float, default=0.5)
	ap.add_argument("--ell_hat", type=float, default=1.5)
	ap.add_argument("--sf2_hat", type=float, default=1.5)
	ap.add_argument("--sxi2_hat", type=float, default=0.2)
	'''
    Data size configuration
    '''
	ap.add_argument("--log10N_step", type=int, required=True)
	ap.add_argument("--step_max", type=int, required=True)
	ap.add_argument("--current_step", type=int, required=True)
	'''
	Output files configuration
	'''
	ap.add_argument("--nn_dir", type=str, default="nn_sets")
	ap.add_argument("--nn_prefix", type=str, default=None)
	ap.add_argument("--out_prefix", type=str, default=None)
	args = ap.parse_args()
    
    # --------------------------------------------------------
    # Processing Arguments
    # --------------------------------------------------------
    
	assert args.mode in ["GPnn", "NNGP"], (
    	"--mode must be one of GPnn|NNGP")
    	
	nu = args.nu
	s = args.log10N_step
	smax = args.step_max
	si = args.current_step
	dim = args.dim

    # --------------------------------------------------------
    # k-NN schedule
    # --------------------------------------------------------
	#nu = min(nu, 1.)
	p = 2*(dim*s+nu*(2*s+si-smax))/(s*(dim+2*nu))
	m = np.ceil(np.pow(10,p)).astype(int)
	print("No. of NNs: ", m)
	#nu = args.nu
	
	nn_file_tag = (
	f"exactNN_{args.distro}_d{dim}_k100_seed{args.seed_train}_"
	f"N1e{si}div{s}_vecs"
	)
	X_query_tag = (
	f"X_query_{args.distro}_d{dim}"
	)
	print("Loading: ", nn_file_tag)
    
    
    # ------------------------------------------------------------
    # Load test points and local training points from file
    # Process the test/train points
    # ------------------------------------------------------------
	X_test = np.load(os.path.join(args.nn_dir, f"{X_query_tag}.npy")) 
	X_train_local = np.load(os.path.join(args.nn_dir, f"{nn_file_tag}.npy"))     
	X_test = X_test / np.sqrt(dim) # normalize
	X_train_local = X_train_local / np.sqrt(dim) # normalise
    
    # select the m-NNs
	nn_filtered = []
	for x, X_nn in zip(X_test, X_train_local):
		idx, _ = knn(x, X_nn, k=m)
		X_nn = np.asarray(X_nn)[idx]
		nn_filtered.append(X_nn)
	X_train_local = np.asarray(nn_filtered)
    
	
	# --------------------------------------------------------
    # Run predictions + calibration
    # --------------------------------------------------------
	n_cal = 2000
	X_cal = X_test[-n_cal:]
	X_cal_local = X_train_local[-n_cal:]
	X_test = X_test[:-n_cal]
	X_train_local = X_train_local[:-n_cal]
	print(f"Running calibration and predictions on {len(X_cal)}:{len(X_test)} CAL:TEST split...")
	
	chunk_size = 2048
	
	rng_xi = np.random.default_rng(seed=125)
	
	# calibration on the CAL-split
	results_cal = run_experiment(
		mode = args.mode,
		X_test = X_cal, 
		X_train_local = X_cal_local,
		nu = nu,
		ell = args.ell,
		sigma_f2=args.sf2,
		sigma_xi2=args.sxi2,
		ell_hat = args.ell_hat,
		b1_hat = args.b1_hat,
		b2_hat = args.b2_hat,
		sigma_f2_hat=args.sf2_hat,
		sigma_xi2_hat=args.sxi2_hat,
		rng_seed = 123, 
		chunk_size=chunk_size,
		return_var=True
		)
	y_cal = results_cal["f_test_true"] + np.sqrt(0.1) * rng_xi.normal(size=results_cal["f_test_true"].shape)
	alpha = np.mean((y_cal - results_cal["mu_pred"])**2/results_cal["var_pred"])
	nll0  = np.mean(np.log(results_cal["var_pred"]) + alpha + np.log(2.*np.pi))/2.
	print("CAL, NLL before calibration: ", alpha, nll0)
	
	# predictions using the calibrated
	# sigma_f2_hat, sigma_xi2_hat
	
	results = run_experiment(
		mode = args.mode,
		X_test = X_test, 
		X_train_local = X_train_local,
		nu = nu,
		ell = args.ell,
		sigma_f2=args.sf2,
		sigma_xi2=args.sxi2,
		ell_hat = args.ell_hat,
		b1_hat = args.b1_hat,
		b2_hat = args.b2_hat,
		sigma_f2_hat=args.sf2_hat * alpha,
		sigma_xi2_hat=args.sxi2_hat * alpha,
		rng_seed = 124, 
		chunk_size=chunk_size,
		return_var=True
		)
	y_test = results["f_test_true"] + np.sqrt(0.1) * rng_xi.normal(size=results["f_test_true"].shape)
	cal = np.mean((y_test - results["mu_pred"])**2/results["var_pred"])
	nll  = np.mean(np.log(results["var_pred"]) + cal + np.log(2.*np.pi))/2.
	print("CAL, NLL after calibtation: ", cal, nll)
	print("mse_vs_latent =", results["mse_vs_latent"])
	
	out_tag = (
    	f"{args.mode}_{args.distro}_d{dim}_seed{args.seed_train}_"
		f"N1e{si}div{s}_nu{nu}"
	)
	
	if args.mode == "GPnn":
		out_dir = "GPnn_results"
	else: 
		out_dir = "NNGP_results"
	os.makedirs(out_dir, exist_ok=True)
	np.save(os.path.join(out_dir, f"{out_tag}_mse.npy"), results["mse_vs_latent"])
	np.save(os.path.join(out_dir, f"{out_tag}_cal.npy"), [[alpha,nll0],[cal,nll]])
	print("Saved to ", os.path.join(out_dir, f"{out_tag}_mse.npy"))
	
	