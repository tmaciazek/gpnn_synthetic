import numpy as np
import resource
import random
import copy
import os
import argparse
import time

from utils import *

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
    
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--distro", type=str, default='gaussian', help="Data distribution: gaussian|uniform_disk")
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--seed_train", type=int, default=0)
    ap.add_argument("--log10_tot_size", type=int, required=True)
    ap.add_argument("--out_dir", type=str, default="nn_sets")
    ap.add_argument("--out_prefix", type=str, default=None)
    args = ap.parse_args()
    
    '''
    	Process arguments
    '''
    assert args.distro in ['gaussian', 'uniform_disk'], ( 
    	"--distro must be one of gaussian|uniform_disk")
    
    device = _select_device('cpu')
    
    '''
    	RNG setup
	'''
    rng_train = np.random.RandomState(args.seed_train)
        
    if args.distro == 'gaussian':
    	sample_train = sample_gaussian(rng=rng_train, device=device)
    	print("Sampling from Gaussian distro.")
    else:
    	sample_train = sample_uniform_disk(rng=rng_train, device=device)
    	print("Sampling from uniform_disk distro.")

    ind_lists = np.load(os.path.join(args.out_dir, f"{args.out_prefix}_inds.npy"))
    
    '''
    	Check for NN duplicates
    '''
    k = ind_lists.shape[1]
    ind_filtered = [dedupe([ind for ind in inds if ind > -1]) for inds in ind_lists]
    full_nn_mask = [len(set(inds)) >= k for inds in ind_filtered]
    assert sum(full_nn_mask) == len(ind_lists), "NN duplicates found!"
	
    '''
		NN retrieval
    '''

    # ------------------------------------------------------------
    # Build flattened mapping: (train_idx, query_idx, slot_idx)
    # slot_idx preserves the original neighbour position 0,...,k-1
    # ------------------------------------------------------------
    nq, k = ind_lists.shape

    train_idx = ind_lists.reshape(-1).astype(np.int64, copy=False)
    query_idx = np.repeat(np.arange(nq, dtype=np.int64), k)
    slot_idx  = np.tile(np.arange(k, dtype=np.int64), nq)

    order = np.argsort(train_idx, kind="stable")
    train_idx = train_idx[order]
    query_idx = query_idx[order]
    slot_idx = slot_idx[order]

    train_size = int(np.ceil(10 ** args.log10_tot_size) + 100_000)
    train_batch_size = min(train_size, 10**6)
    train_batches = int(np.ceil(train_size / train_batch_size))

    nn_x = None
    tot_data = 0

    for n_batch in range(train_batches):
        print("train_batches, n_batch:", train_batches, n_batch, flush=True)

        curr_bs = min(train_batch_size, train_size - tot_data)
        if curr_bs <= 0:
            break

        X = sample_train(size=(curr_bs, args.dim))

        ind_min = tot_data
        ind_max = tot_data + curr_bs

        # Since train_idx is sorted, find the relevant slice in O(log(nq*k))
        lo = np.searchsorted(train_idx, ind_min, side="left")
        hi = np.searchsorted(train_idx, ind_max, side="left")

        print("No. of nn train indices in this batch:\t", hi - lo, flush=True)

        if lo < hi:
            local_idx = train_idx[lo:hi] - ind_min
            q_idx = query_idx[lo:hi]
            s_idx = slot_idx[lo:hi]

            # fetch repeated train rows only once within the batch
            uniq_local, inv = np.unique(local_idx, return_inverse=True)

            if torch.is_tensor(X):
                uniq_local_t = torch.from_numpy(uniq_local).long().to(X.device)
                rows = X.index_select(0, uniq_local_t)

                if nn_x is None:
                    nn_x = torch.empty((nq, k, X.shape[1]), dtype=X.dtype, device=X.device)

                q_t = torch.from_numpy(q_idx).long().to(X.device)
                s_t = torch.from_numpy(s_idx).long().to(X.device)
                inv_t = torch.from_numpy(inv).long().to(X.device)

                nn_x[q_t, s_t] = rows.index_select(0, inv_t)

            else:
                rows = X[uniq_local]

                if nn_x is None:
                    nn_x = np.empty((nq, k, X.shape[1]), dtype=X.dtype)

                nn_x[q_idx, s_idx] = rows[inv]

        print(
            "MEM usage:\t"
            + str(int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0**2)))
            + " GB"
        )
        print()

        tot_data += curr_bs


    assert nn_x.shape[:2] == (nq, k)
    np.save(os.path.join(args.out_dir, f"{args.out_prefix}_vecs.npy"), nn_x)
    print(f"Saved {args.out_prefix}_vecs")
    
    
if __name__ == "__main__":
    main()
    
