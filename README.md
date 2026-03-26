# The Theory and Practice of Highly Scalable Gaussian Process Regression with Nearest Neighbours

This repository is the official implementation of synthetic-data experiments from [The Theory and Practice of Highly Scalable Gaussian Process Regression with Nearest Neighbours](https://arxiv.org/).


 ### 1. (Optional) Sample training/test data and find nearest-neighbours.
 This is optional. The pre-sampled test data and pre-calculated corresponding *indices* of nearest neighbours are contained in the folder [nn_sets](nn_sets). If you do want to re-generate this, run the following commands.

```
python generate_query_data.py --dim 2 --data_size 10000  --distro uniform_disk
for seed in 0 1 2 3; do
    python exact_nn.py --dim 2 --q_data_size 10000 --batch_size 1000 --num_batches 1000 --q_bs 2048 --seed_train $seed --device cpu --distro uniform_disk --checkpoint_freq 4
done
```
To re-generate the Gaussian data used in the $GPnn$-experiments in dimension $d_X=4$, run the following commands. As opposed to the above $d_X=2$-case, the scripts below require a GPU. The `bash` for-loop can be replaced by an array-job when submitting the jobs on Slurm.
```
python generate_query_data.py --dim 4 --data_size 10000  --distro gaussian
for seed in 0 1 2 3; do
    python exact_nn.py --dim 4 --q_data_size 10000 --batch_size 200000 --num_batches 2812000 --q_bs 2048 --seed_train $seed --device gpu --distro gaussian --checkpoint_freq 2
done
```
For higher dimensions, replace `--dim 4` with `--dim 8` or `--dim 16`. The runtime grows quickly with the dimension (for $d_X=16$ it took about 80 hours on NVIDIA Tesla V100).
