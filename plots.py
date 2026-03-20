import numpy as np
import sys
import os

from scipy.interpolate import CubicSpline

import matplotlib.pyplot as plt

dim = 2
#nu = float(sys.argv[2])
sigma_xi2 = 1e-10 + 0.1
m = 100

out_dir = "NNGP_results"

nus = [0.5,0.75,1.5]
nus_txt = ['1/2','3/4','3/2','5/2']
seeds = [0]
s_iter = range(13, 25, 1)
s = 4

for nu, nu_txt in zip(nus, nus_txt):
	mse_list = []
	for si in s_iter:
		mse = 0.
		for seed_train in seeds:
			out_tag = (
    			#f"GPnn_mvar_d{dim}_"
    			#f"N{twologN}div2_seed{seed_train}_nu{nu}"
    			f"NNGP_uniform_disk_d{dim}_seed{seed_train}_"
    			f"N1e{si}div4_nu{nu}_mse"
			)
	
			mse += np.load(os.path.join(out_dir, f"{out_tag}.npy"))
		mse_list.append(mse.squeeze()/len(seeds))

	mse_list = np.asarray(mse_list)
	log_mse_list = np.log10(mse_list)

	logN_list = np.asarray(list(s_iter))/s #change

	z = np.polyfit(logN_list[-7:], log_mse_list[-7:], 1)
	p = np.poly1d(z)
	print("Nu, Fitted slope:", nu, z[0])
	print("Theory:", 2 * nu/(dim+2 * nu))
	
	# R^2
	ss_res = np.sum((log_mse_list[-8:] - p(logN_list)[-8:])**2)
	ss_tot = np.sum((log_mse_list[-8:] - np.mean(log_mse_list[-8:]))**2)
	r2 = 1 - ss_res / ss_tot

	print("R2: ", r2)

	plt.plot(logN_list, p(logN_list), linestyle='--')
	plt.scatter(logN_list, log_mse_list, label=r'$\nu$='+nu_txt)

plt.xlabel(r'$\log_{10}n$',fontsize=20)
plt.ylabel(r'$\log_{10}\widehat{\mathcal{R}}_n$',fontsize=20)
plt.legend(fontsize=15)
ax = plt.gca()
ax.set_box_aspect(2/3)
plt.tight_layout()
plt.show()


'''
nu = 0.5
seeds = [0,1,2,3,4]
ell_list = np.linspace(0.5,2.5,40)
sf_list = np.linspace(1.,4.0,40)
sxi_list = np.linspace(0.05, 0.3, 40)
b_list = np.linspace(0.0, 2.0, 40)
#twologN_list = [12,17,22]
twologN_list = [16,20,24]
y_list = []
for twologN in twologN_list:
	mse_list = []
	for par_ind in range(40):
		mse = 0.
		for seed_train in seeds:
			out_tag = (
    			f"nngp_mvar_d{dim}_"
    			f"N{twologN}div4_seed{seed_train}_nu{nu}_b{par_ind}_landscape" #change
    			#f"N{twologN}div2_seed{seed_train}_nu{nu}_sxi{par_ind}_landscape"
			)
	
			mse += np.load(os.path.join(out_dir, f"{out_tag}_mse.npy"))
		mse_list.append(mse.squeeze()/len(seeds))

	mse_list = np.asarray(mse_list)
	print(mse_list)
	y_list.append(mse_list)
#mse_list -=  sigma_xi2/m
#print(mse_list)

for mse_list,twologN in zip(y_list, twologN_list):
	mse_list = mse_list-mse_list[-1]
	print(int(twologN/4) - twologN/4.)
	if abs(int(twologN/4) - twologN/4.) < 1e-7:
		logsize = int(twologN/4)
	else:
		logsize = twologN/4
	plt.plot(b_list, mse_list, label=rf'$n=1e{logsize}$') #change

plt.legend(fontsize=15)
#plt.xlabel(r'$\hat\ell$',fontsize=20)

 
plt.ylabel(r'$\widehat{\mathcal{R}}_n$'+' (shifted)',fontsize=20)
plt.tight_layout()
plt.show()
'''
'''
nu = 0.5
labels = [r'\hat\ell', r'\hat\sigma_f^2', r'\hat\sigma_f^2', r'\hat\boldsymbol{b}']
tags = ['ell', 'sf', 'sxi','b']
seeds = [0,1,2,3,4]
twologN_iter = range(13, 24, 1)

dl=0.01
d_coeffs = np.asarray([1., -8., 0, 8., -1.])/(12.*dl)

for label, tag in zip(labels,tags):
	der_list = []
	for twologN in twologN_iter:
		mse_list = []
		for par_ind in range(5):
			mse = 0.
			for seed_train in seeds:
				out_tag = (
    				f"GPnn_mvar_d{dim}_"
    				f"N{twologN}div4_seed{seed_train}_nu{nu}_"+tag+f"{par_ind}"
				)
	
				mse += np.load(os.path.join(out_dir, f"{out_tag}_mse.npy"))
			mse = mse.squeeze()/len(seeds)
			mse_list.append(mse.squeeze()/len(seeds))
		mse_list = np.asarray(mse_list)
		der_list.append(np.sum(mse_list*d_coeffs))


	der_list = np.asarray(der_list)
	log_der_list = np.log10(np.abs(der_list))
	logN_list = np.asarray(list(twologN_iter))/4.

	z = np.polyfit(logN_list[-8:], log_der_list[-8:], 1)
	p = np.poly1d(z)
	
	print(label)
	print("Fitted slope:", z[0])
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
plt.show()
'''
