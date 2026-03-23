import numpy as np
#import matplotlib.pyplot as plt
import sys

import numpy as np

for s in range(13,25):
	results_before = []
	results_after = []
	print(s)
	for seed in range(0,4):
		result = np.load(f"NNGP_results/NNGP_uniform_disk_d2_seed{seed}_N1e{s}div4_nu0.5_cal.npy")
		results_before.append(result[0])
		results_after.append(result[1])
	results_before = np.row_stack(results_before)
	results_after = np.row_stack(results_after)
	print(results_before.mean(0), results_before.std(0))
	print(results_after.mean(0), results_after.std(0))

'''
X = np.load("nn_sets/X_query_uniform_disk_d2.npy")
print(X.shape)

d = np.load("nn_sets/exactNN_uniform_disk_d2_k100_seed0_N1e16div4_dists.npy")
d2 = np.load("nn_sets/exactNN_uniform_disk_d2_k100_seed0_N1e16div4_dists2.npy")
v = np.load("nn_sets/exactNN_uniform_disk_d2_k100_seed0_N1e16div4_vecs.npy")

print(np.mean(np.abs(d-d2)))
print('\n')
print(X.shape)
'''
'''
plt.scatter([X[0][0]], [X[0][1]])
plt.scatter(v[0][:,0], v[0][:,1])
plt.show()
'''