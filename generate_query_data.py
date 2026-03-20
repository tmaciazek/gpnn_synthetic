#!/usr/bin/env python3
import argparse
import os
import numpy as np
import torch

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
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--distro", type=str, default='gaussian', help="Data distribution: gaussian|uniform_disk")
    ap.add_argument("--float_type", type=int, default=64, help="32|64")
    ap.add_argument("--data_size", type=int, default=10_000, help="Only used with --synthetic_query")
    ap.add_argument("--seed_query", type=int, default=1234)
    ap.add_argument("--data_gen", type=str, default="numpy", help="numpy|torch")
    ap.add_argument("--out_dir", type=str, default="nn_sets")
    ap.add_argument("--out_prefix", type=str, default=None)
    args = ap.parse_args()
    
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
    
    device = _select_device('cpu')
    
    '''
    	RNG setup
	'''
    if args.data_gen == "numpy":
    	rng_q = np.random.RandomState(args.seed_query)
    else:
        rng_q = torch.Generator(device=device)
        rng_q.manual_seed(args.seed_query)
        
    if args.distro == 'gaussian':
    	sample_q = sample_gaussian(rng=rng_q, gen=args.data_gen, num_type=num_type, device=device)
    	print("Sampling from Gaussian distro.")
    else:
    	sample_q = sample_uniform_disk(rng=rng_q, gen=args.data_gen, num_type=num_type, device=device)
    	print("Sampling from uniform_disk distro.")
	
    '''
		Generate query data
    '''
    X_query = sample_q(size=(args.data_size,args.dim))
    '''
		Preparation for saving data
	'''
    os.makedirs(args.out_dir, exist_ok=True)
    if args.out_prefix is not None:
    	out_prefix = args.out_prefix
    else:
    	out_prefix = f"X_query_{args.distro}_d{args.dim}"

    np.save(os.path.join(args.out_dir, f"{out_prefix}.npy"), X_query.cpu().numpy())
	
if __name__ == "__main__":
    main()
