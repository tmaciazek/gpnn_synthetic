import numpy as np
import sys
import os
import argparse
from fractions import Fraction

import matplotlib.pyplot as plt

def float_to_fraction_string(x, max_denominator=1000000):
    return str(Fraction(x).limit_denominator(max_denominator))

ap = argparse.ArgumentParser()
ap.add_argument("--mode", type=str, default='NNGP', help="Regression mode: GPnn|NNGP")
ap.add_argument("--dims", nargs="+", type=int, default=[2])
ap.add_argument("--seeds", nargs="+", type=int, default=[0])
ap.add_argument("--nus", nargs="+", type=float, default=[0.5])
args = ap.parse_args()
	

seeds = args.seeds
#mode = "GPnn"
if args.mode == "NNGP":
	distro = "uniform_disk"
	s_iter = range(13, 25, 1)
	s = 4
	dim = args.dims[0]
	pars = args.nus
	pars_txt = [float_to_fraction_string(nu) for nu in pars]
else:
	distro = "gaussian"
	s_iter = range(12, 24, 1)
	s = 2
	nu = args.nus[0]
	pars = args.dims
	pars_txt = [float_to_fraction_string(dim) for dim in pars]

out_dir = f"{args.mode}_results"

for par, par_txt in zip(pars, pars_txt):
	mse_list = []
	for si in s_iter:
		mse = 0.
		for seed_train in seeds:
			if args.mode == "NNGP":
				out_tag = (
    				f"{args.mode}_{distro}_d{dim}_seed{seed_train}_"
    				f"N1e{si}div{s}_nu{par}_mse"
				)
			else:
				out_tag = (
    				f"{args.mode}_{distro}_d{par}_seed{seed_train}_"
    				f"N1e{si}div{s}_nu{nu}_mse"
				)
	
			mse += np.load(os.path.join(out_dir, f"{out_tag}.npy"))
		mse_list.append(mse.squeeze()/len(seeds))

	mse_list = np.asarray(mse_list)
	log_mse_list = np.log10(mse_list)

	logN_list = np.asarray(list(s_iter))/s #change

	z = np.polyfit(logN_list[-7:], log_mse_list[-7:], 1)
	p = np.poly1d(z)
	if args.mode == "NNGP":
		print("Nu, Fitted slope:", par, z[0])
		print("Theory:", 2 * par/(dim+2 * par))
	
		# R^2
		ss_res = np.sum((log_mse_list[-8:] - p(logN_list)[-8:])**2)
		ss_tot = np.sum((log_mse_list[-8:] - np.mean(log_mse_list[-8:]))**2)
		r2 = 1 - ss_res / ss_tot

		print("R2: ", r2)

		plt.plot(logN_list, p(logN_list), linestyle='--')
		plt.scatter(logN_list, log_mse_list, label=r'$\nu$='+par_txt)
	else:
		print("Fim, Fitted slope:", par, z[0])
		print("Theory:", 2 * nu/(par+2 * nu))
	
		# R^2
		ss_res = np.sum((log_mse_list[-8:] - p(logN_list)[-8:])**2)
		ss_tot = np.sum((log_mse_list[-8:] - np.mean(log_mse_list[-8:]))**2)
		r2 = 1 - ss_res / ss_tot

		print("R2: ", r2)

		plt.plot(logN_list, p(logN_list), linestyle='--')
		plt.scatter(logN_list, log_mse_list, label=r'$d_X$='+par_txt)
		

plt.xlabel(r'$\log_{10}n$',fontsize=20)
plt.ylabel(r'$\log_{10}\widehat{\mathcal{R}}_n$',fontsize=20)
plt.legend(fontsize=15)
ax = plt.gca()
ax.set_box_aspect(2/3)
plt.tight_layout()
plt.show()
