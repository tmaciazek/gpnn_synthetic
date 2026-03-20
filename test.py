import numpy as np
import matplotlib.pyplot as plt
import sys

import numpy as np

import numpy as np

X = np.load("nn_sets/X_query_uniform_disk_d2.npy")
print(X.shape)

d = np.load("nn_sets/exactNN_uniform_disk_d2_k100_seed0_N1e16div4_dists.npy")
d2 = np.load("nn_sets/exactNN_uniform_disk_d2_k100_seed0_N1e16div4_dists2.npy")
v = np.load("nn_sets/exactNN_uniform_disk_d2_k100_seed0_N1e16div4_vecs.npy")

print(np.mean(np.abs(d-d2)))
print('\n')
print(X.shape)
'''
plt.scatter([X[0][0]], [X[0][1]])
plt.scatter(v[0][:,0], v[0][:,1])
plt.show()
'''