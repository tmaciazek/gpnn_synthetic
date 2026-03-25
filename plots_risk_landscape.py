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


ell_list = np.linspace(0.5,2.5,40)
sf_list = np.linspace(1.,4.0,40)
sxi_list = np.linspace(0.05, 0.3, 40)
b_list = np.linspace(0.0, 2.0, 40)
si_list = [16,20,24]

for par_id in ['ell', 'sf2', 'sxi2', 'b']:

	if par_id == 'ell':
		par_list = ell_list
	elif par_id == 'sf2':
		par_list = sf_list
	elif par_id == 'sxi2':
		par_list = sxi_list
	elif par_id == 'b':
		par_list = b_list

	y_list = []
	for si in si_list:
		mse_list = []
		for par_ind in range(40):
			mse = 0.
			for seed_train in seeds:
				out_tag = (
					f"{args.mode}_{distro}_d{dim}_seed{seed_train}_"
					f"N1e{si}div{s}_nu{nu}_landscape"
				)
				mse += np.load(os.path.join(out_dir, f"{out_tag}_{par_id}_{par_ind}.npy"))
			mse_list.append(mse.squeeze()/len(seeds))

		mse_list = np.asarray(mse_list)
		print(mse_list)
		y_list.append(mse_list)
	plt.clf()
	for mse_list,si in zip(y_list, si_list):
		mse_list = mse_list-mse_list[-1]
		logsize = int(si/s)
		plt.plot(par_list, mse_list, label=rf'$n=1e{logsize}$') #change

	plt.legend(fontsize=15)
	plt.ylabel(r'$\widehat{\mathcal{R}}_n$'+' (shifted)',fontsize=20)
	if par_id == 'ell':
		plt.xlabel(r'$\hat\ell$',fontsize=20)
	elif par_id == 'sf2':
		plt.xlabel(r'$\hat\sigma_f^2$',fontsize=20)
	elif par_id == 'sxi2':
		plt.xlabel(r'$\hat\sigma_\xi^2$',fontsize=20)
	elif par_id == 'b':
		plt.xlabel(r'$\hat b$',fontsize=20)
	plt.tight_layout()
	plt.savefig(os.path.join(plots_dir, f"{args.mode}_landscape_{par_id}.pdf"))

'''

for par, par_txt in zip(pars, pars_txt):
	mse_list = []
	for si in s_iter:
		mse = 0.
		for seed_train in seeds:
			if args.mode == "NNGP":
				out_tag = (
    				f"{args.mode}_{distro}_d{dim}_seed{seed_train}_"
    				f"N1e{si}div{s}_nu{par}_risk"
				)
			else:
				out_tag = (
    				f"{args.mode}_{distro}_d{par}_seed{seed_train}_"
    				f"N1e{si}div{s}_nu{nu}_risk"
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
		ss_res = np.sum((log_mse_list[-10:] - p(logN_list)[-10:])**2)
		ss_tot = np.sum((log_mse_list[-10:] - np.mean(log_mse_list[-10:]))**2)
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
'''