# The Theory and Practice of Highly Scalable Gaussian Process Regression with Nearest Neighbours

This repository is the official implementation of synthetic-data experiments from [The Theory and Practice of Highly Scalable Gaussian Process Regression with Nearest Neighbours](https://arxiv.org/).


 ### 1. (Optional) Sample training/test data and find nearest-neighbours.
 This is optional. The pre-sampled test data and pre-calculated corresponding *indices* of nearest neighbours. This is contained in the folder [nn_sets](nn_sets). If you do want to re-calculate this, run the following commands.

```
python generate_query_data.py --dim 2 --data_size 10000  --distro uniform_disk
python exact_nn.py --dim 2 --q_data_size 10000 --batch_size 1000 --num_batches 1000 --q_bs 2048 --seed_train 0 --device cpu --distro uniform_disk --checkpoint_freq 4
python exact_nn.py --dim 2 --q_data_size 10000 --batch_size 1000 --num_batches 1000 --q_bs 2048 --seed_train 1 --device cpu --distro uniform_disk --checkpoint_freq 4
python exact_nn.py --dim 2 --q_data_size 10000 --batch_size 1000 --num_batches 1000 --q_bs 2048 --seed_train 2 --device cpu --distro uniform_disk --checkpoint_freq 4
python exact_nn.py --dim 2 --q_data_size 10000 --batch_size 1000 --num_batches 1000 --q_bs 2048 --seed_train 3 --device cpu --distro uniform_disk --checkpoint_freq 4
```
