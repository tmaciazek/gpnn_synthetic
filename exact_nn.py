#!/usr/bin/env python3
"""
Exact k-NN (Euclidean) by brute force, streaming X_train batch-by-batch.

Rewrite of the attached exact_nn.py to use PyTorch. If --device gpu, computation runs on CUDA.

- Outer loop: training batches (stream)
- Inner loop: query blocks (to keep RAM bounded)
- Exact distances, exact top-k

Notes:
- Default: generate X_train batches with NumPy RNG (matches your earlier setup), then move to torch device.
- Optional: --train_gen torch generates train batches directly on the target device (faster, but different RNG stream vs NumPy).

Example:
  python exact_nn_torch.py --synthetic_query --dim 8 --q_data_size 10000 \
    --k 100 --batch_size 100000 --num_batches 1000 --q_bs 1000 \
    --seed_train 42 --seed_query 43 --device gpu --train_gen torch
"""

import argparse
import os
import numpy as np
import torch

def sample_uniform_disk(radius: float = 1.0, center=(0.0, 0.0), *, rng=None, gen: str = "numpy", num_type=torch.float32, device: torch.device | None = None):
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



def _select_device(device_arg: str) -> torch.device:
    device_arg = device_arg.lower()
    if device_arg in ("auto", "cuda", "gpu"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if device_arg in ("cuda", "gpu"):
            raise RuntimeError("--device gpu requested but CUDA is not available.")
        return torch.device("cpu")
    if device_arg == "cpu":
        return torch.device("cpu")
    raise ValueError("--device must be one of: auto, cpu, gpu")


@torch.no_grad()
def exact_knn_stream(
    X_query: torch.Tensor,
    train_sampling_fn,
    k: int,
    dim: int,
    batch_size: int,
    num_batches: int,
    q_bs: int = 512,
    train_index_offset: int = 0,
    device: torch.device | None = None,
    train_gen: str = "numpy",
    out_dir = None,
    out_prefix = None,
    checkpoint_freq = 1,
    num_type = torch.float64
):

    X_query = X_query.contiguous()
    M, d = X_query.shape
    assert d == dim, f"Query dim {d} != dim {dim}"

    # Precompute ||q||^2 once (queries reused for every train batch)
    query_norm2_all = (X_query * X_query).sum(dim=1)  # (M,)

    # Global best for each query across all streamed train batches
    best_d2_all = torch.full((M, k), float("inf"), device=device, dtype=num_type)
    best_ix_all = torch.full((M, k), -1, device=device, dtype=torch.long)
    
    tot_size = 0

    for b in range(num_batches):
        T = train_sampling_fn(size=(batch_size, dim))

        tb = T.shape[0]
        tn2 = (T * T).sum(dim=1)  # (tb,)

        for q0 in range(0, M, q_bs):
            q1 = min(M, q0 + q_bs)
            Q = X_query[q0:q1]            # (qb, d)
            qn2 = query_norm2_all[q0:q1]  # (qb,)
            qb = Q.shape[0]

            # squared distances: ||q - t||^2 = ||q||^2 + ||t||^2 - 2 q·t
            d2 = qn2[:, None] + tn2[None, :] - 2.0 * (Q @ T.t())
            d2.clamp_(min=0.0)

            # Best k within this train batch for each query in the block
            kk = k if tb >= k else tb
            if tb > kk:
                cand_d2, cand_j = torch.topk(d2, kk, dim=1, largest=False, sorted=False)
            else:
                cand_j = torch.argsort(d2, dim=1)[:, :kk]
                cand_d2 = torch.gather(d2, 1, cand_j)

            cand_ix = cand_j.to(torch.long) + (tot_size + train_index_offset)

            # Merge with current best for this query block (2k -> keep k)
            best_d2 = best_d2_all[q0:q1]
            best_ix = best_ix_all[q0:q1]

            if kk < k:
                pad = k - kk
                pad_d2 = torch.full((qb, pad), float("inf"), device=device, dtype=num_type)
                pad_ix = torch.full((qb, pad), -1, device=device, dtype=torch.long)
                cand_d2 = torch.cat([cand_d2, pad_d2], dim=1)
                cand_ix = torch.cat([cand_ix, pad_ix], dim=1)

            merged_d2 = torch.cat([best_d2, cand_d2], dim=1)  # (qb, 2k)
            merged_ix = torch.cat([best_ix, cand_ix], dim=1)  # (qb, 2k)

            new_d2, new_pos = torch.topk(merged_d2, k, dim=1, largest=False, sorted=False)
            new_ix = torch.gather(merged_ix, 1, new_pos)

            best_d2_all[q0:q1] = new_d2
            best_ix_all[q0:q1] = new_ix

        tot_size += batch_size
        if (b + 1) % max(1, num_batches // 100) == 0:
            print(f"Training batch {b+1}/{num_batches}", flush=True)
        
        # Checkpoint saving
        if tot_size > batch_size:
        	p1 = int(checkpoint_freq*np.log10(tot_size-batch_size))
        	p2 = int(checkpoint_freq*np.log10(tot_size))
        	if p2-p1 == 1:
        		best_d2_sorted, order = torch.sort(best_d2_all, dim=1, descending=False)
        		best_ix_sorted = torch.gather(best_ix_all, 1, order)
        		best_dists = torch.sqrt(best_d2_sorted)
        	
        		inds = best_ix_sorted.cpu().numpy().astype(np.int64, copy=False)
        		dists = best_dists.cpu().numpy()

        		out_name = out_prefix + f"_N1e{p2}div"+str(checkpoint_freq)
        		np.save(os.path.join(out_dir, f"{out_name}_inds.npy"), inds)
        		np.save(os.path.join(out_dir, f"{out_name}_dists.npy"), dists)
        		print("Saved:", os.path.join(out_dir, f"{out_name}_inds.npy"), os.path.join(out_dir, f"{out_name}_dists.npy"),flush=True)
        	

    best_d2_sorted, order = torch.sort(best_d2_all, dim=1, descending=False)
    best_ix_sorted = torch.gather(best_ix_all, 1, order)
    best_dists = torch.sqrt(best_d2_sorted)

    return (
        best_ix_sorted.cpu().numpy().astype(np.int64, copy=False),
        best_dists.cpu().numpy(),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--distro", type=str, default='gaussian', help="Data distribution: gaussian|uniform_disk")
    ap.add_argument("--float_type", type=int, default=64, help="32|64")
    ap.add_argument("--q_data_size", type=int, default=10_000, help="Only used with --synthetic_query")
    ap.add_argument("--k", type=int, required=True, help="No. of NNs")
    ap.add_argument("--seed_query", type=int, default=43)
    ap.add_argument("--seed_train", type=int, default=0)
    ap.add_argument("--batch_size", type=int, required=True)
    ap.add_argument("--num_batches", type=int, required=True)
    ap.add_argument("--q_bs", type=int, default=512, help="Query set batch size")
    ap.add_argument("--train_index_offset", type=int, default=0)
    ap.add_argument("--device", type=str, default="auto", help="auto|cpu|gpu")
    ap.add_argument("--data_gen", type=str, default="numpy", help="numpy|torch")
    ap.add_argument("--out_dir", type=str, default="nn_sets")
    ap.add_argument("--out_prefix", type=str, default=None)
    ap.add_argument("--checkpoint_freq", type=int, required=True, help="Checkpoint frequency.")
    args = ap.parse_args()

    device = _select_device(args.device)
    print("CUDA available?\t", torch.cuda.is_available(), flush=True)
    print("Using device:\t", device, flush=True)
    
    '''
    	Process arguments
    '''
    assert args.float_type in [32, 64], (
    	"--float_type must be one of 32|64")
    assert args.distro in ['gaussian', 'uniform_disk'], ( 
    	"--distro must be one of gaussian|uniform_disk")
    assert args.data_gen in ['numpy', 'torch'], (
    	"--data_gen must be one of numpy|torch")
    
    if args.float_type == 32: num_type = torch.float32
    else: num_type = torch.float64
    
    '''
    	RNG setup
	'''
    if args.data_gen == "numpy":
    	rng_train = np.random.RandomState(args.seed_train)
    	rng_q = np.random.RandomState(args.seed_query)
    else:
        rng_train = torch.Generator(device=device)
        rng_train.manual_seed(args.seed_train)
        rng_q = torch.Generator(device=device)
        rng_q.manual_seed(args.seed_query)
        
    if args.distro == 'gaussian':
    	sample_train = sample_gaussian(rng=rng_train, gen=args.data_gen, num_type=num_type, device=device)
    	sample_q = sample_gaussian(rng=rng_q, gen=args.data_gen, num_type=num_type, device=device)
    else:
    	sample_train = sample_uniform_disk(rng=rng_train, gen=args.data_gen, num_type=num_type, device=device)
    	sample_q = sample_uniform_disk(rng=rng_q, gen=args.data_gen, num_type=num_type, device=device)
	
    '''
		Generate query data
    '''
    X_query = sample_q(size=(args.q_data_size,args.dim))
    print(X_query)
    '''
		Preparation for saving checkpoints
	'''
    os.makedirs(args.out_dir, exist_ok=True)
    if args.out_prefix is not None:
    	out_prefix = args.out_prefix
    else:
    	out_prefix = f"exactNN_{args.distro}_d{args.dim}_seed{args.seed_train}"

    inds, dists = exact_knn_stream(
        X_query=X_query,
        train_sampling_fn=sample_train,
        k=args.k,
        dim=args.dim,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        q_bs=args.q_bs,
        train_index_offset=args.train_index_offset,
        device=device,
        out_dir = args.out_dir,
        out_prefix = out_prefix,
        checkpoint_freq = args.checkpoint_freq,
        num_type = num_type
    )
	
if __name__ == "__main__":
    main()
