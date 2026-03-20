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
    
    all_ind_tuples = [
    	(ind_train, ind_q)
    	for ind_q in range(len(ind_lists))
    	for ind_train in ind_lists[ind_q]
    ]
    
    tic = time.perf_counter()
    all_ind_tuples = np.array(sorted(all_ind_tuples), dtype=np.int64).reshape(-1, 2)
    #all_ind_tuples = np.array( all_ind_tuples, dtype=np.int64).reshape(-1, 2)
    toc = time.perf_counter()
    print("Sorting pairs took " + str(toc - tic) + " seconds", flush=True)
    
    nn_x_lists = [[] for i in range(len(ind_lists))]
    tot_data = 0
    train_size = np.ceil(np.pow(10,args.log10_tot_size)).astype(int)+1e5
    train_batch_size = min(train_size, 10**6)
    train_batches = np.ceil(train_size / train_batch_size).astype(int)
    
    for n_batch in range(train_batches):
    	print("train_batches, n_batch: ", train_batches, n_batch, flush=True)
    
    	#X = rng_x_train.normal(size=(train_batch_size, dim))
    	X = sample_train(size=(train_batch_size, args.dim))
    
    	ind_min = tot_data
    	ind_max = tot_data + train_batch_size
    	batch_mask = (all_ind_tuples[:, 0] >= ind_min) & (
        	all_ind_tuples[:, 0] < ind_max
    	)
    	print("No. of nn train indices in this batch:\t", sum(batch_mask), flush=True)

    	batch_ind_tuples = all_ind_tuples[batch_mask]
    	batch_ind_tuples[:, 0] = batch_ind_tuples[:, 0] - tot_data
    	#if len(batch_ind_tuples)>0:
    	#	print(min(batch_ind_tuples[:, 0]), max(batch_ind_tuples[:, 0]), flush=True)
    	[
        	nn_x_lists[indp[1]].append(copy.copy(X[indp[0]]))
        	for indp in batch_ind_tuples
    	]
    	print(
        	"MEM usage:\t"
        	+ str(
            	int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0**2))
        	)
        	+ " GB"
    	)
    	print("\n")
        
    	tot_data += train_batch_size
    
    assert sum([len(nn) < k for nn in nn_x_lists]) == 0, "Found non-full NN sets!"
    nn_x_lists = [torch.row_stack(nn) for nn in nn_x_lists]
    nn_x_lists = torch.stack(nn_x_lists).numpy()
    np.save("nn_sets/"+f"{args.out_prefix}_vecs.npy", nn_x_lists)
    print(f"Saved {args.out_prefix}_vecs")
    
    
if __name__ == "__main__":
    main()
    
