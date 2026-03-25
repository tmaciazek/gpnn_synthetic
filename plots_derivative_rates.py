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
ap.add_argument("--dim", type=int, default=2)
ap.add_argument("--seeds", nargs="+", type=int, default=[0])
ap.add_argument("--nu", type=float, default=0.5)
args = ap.parse_args()

seeds = args.seeds
nu = args.nu
dim  = args.dim
if args.mode == "NNGP":
	distro = "uniform_disk"
	s_iter = range(13, 25, 1)
	s = 4
else:
	distro = "gaussian"
	s_iter = range(12, 24, 1)
	s = 2
	
out_dir = f"{args.mode}_results"

plots_dir = f"{args.mode}_plots"
os.makedirs(plots_dir, exist_ok=True)	

labels = [r'\hat\ell', r'\hat\sigma_f^2', r'\hat\sigma_\xi^2', r'\hat b']
tags = ['ell', 'sf2', 'sxi2', 'b']

dl=0.01
d_coeffs = np.asarray([1., -8., 0, 8., -1.])/(12.*dl)

for label, tag in zip(labels,tags):
	der_list = []
	for si in s_iter:
		mse_list = []
		for par_ind in range(5):
			mse = 0.
			for seed_train in seeds:
				out_tag = (
					f"{args.mode}_{distro}_d{dim}_seed{seed_train}_"
					f"N1e{si}div{s}_nu{nu}_D_{tag}_{par_ind}"
				)
	
				mse += np.load(os.path.join(out_dir, f"{out_tag}.npy"))
			mse = mse.squeeze()/len(seeds)
			mse_list.append(mse.squeeze()/len(seeds))
		mse_list = np.asarray(mse_list)
		der_list.append(np.sum(mse_list*d_coeffs))


	der_list = np.asarray(der_list)
	log_der_list = np.log10(np.abs(der_list))
	logN_list = np.asarray(list(s_iter))/4.

	z = np.polyfit(logN_list[-8:], log_der_list[-8:], 1)
	p = np.poly1d(z)
	
	print(label)
	print("Fitted slope:", z[0])
	if tag == 'b':
		print("Theory:", 4 * nu/(dim+2 * nu))
	else:
		print("Theory:", 2 * nu/(dim+2 * nu))

	# R^2
	ss_res = np.sum((log_der_list[-8:] - p(logN_list)[-8:])**2)
	ss_tot = np.sum((log_der_list[-8:] - np.mean(log_der_list[-8:]))**2)
	r2 = 1 - ss_res / ss_tot
	print("R2: ", r2)
	print('\n')

	plt.plot(logN_list, p(logN_list), linestyle='--')
	plt.scatter(logN_list, log_der_list, label=rf'$\phi={label}$')

plt.xlabel(r'$\log_{10}n$',fontsize=20)
plt.ylabel(r'$\log_{10}\left|\partial_\phi\widehat{\mathcal{R}}_n\right|$',fontsize=20)
plt.legend(fontsize=15)
ax = plt.gca()
ax.set_box_aspect(2/3)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, f"{args.mode}_derivative_rates.pdf"))

