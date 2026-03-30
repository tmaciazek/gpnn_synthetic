# The Theory and Practice of Highly Scalable Gaussian Process Regression with Nearest Neighbours

This repository is the official implementation of synthetic-data experiments from [The Theory and Practice of Highly Scalable Gaussian Process Regression with Nearest Neighbours](https://arxiv.org/).

In addition, this repository contains the [Online Appendix 1](Online_Appendix_1.pdf) which displays all the proofs of the theoretical results and provides additinal information about the work.


 ### 1. (Optional) Sample training/test data and find nearest-neighbours.
 This is optional. The pre-sampled test data and pre-calculated corresponding *indices* of nearest neighbours are contained in the folder [nn_sets](nn_sets). If you do want to re-generate this, run the following commands.

```
python generate_query_data.py --dim 2 --data_size 10000  --distro uniform_disk
for seed in 0 1 2 3; do
    python exact_nn.py --dim 2 --q_data_size 10000 --batch_size 1000 --num_batches 1000 --q_bs 2048 --seed_train $seed --device cpu --distro uniform_disk --checkpoint_freq 4
done
```
To re-generate the Gaussian data (training data size $10^{23/2}$) used in the $GPnn$-experiments in dimension $d_X=4$, run the following commands. As opposed to the above $d_X=2$-case, the scripts below require a GPU. The `bash` for-loop can be replaced by an array-job when submitting the jobs on SLURM.
```
python generate_query_data.py --dim 4 --data_size 10000  --distro gaussian
for seed in 0 1 2 3; do
    python exact_nn.py --dim 4 --q_data_size 10000 --batch_size 200000 --num_batches 2812000 --q_bs 2048 --seed_train $seed --device gpu --distro gaussian --checkpoint_freq 2
done
```
For higher dimensions, replace `--dim 4` with `--dim 8` or `--dim 16`. The runtime grows quickly with the dimension (for $d_X=16$ it took about 80 hours on NVIDIA Tesla V100).

 ### 2. Retrieve the nearest-neighbour vectors.

In this step, we retrieve nearest-neighbour vectors from their indices. For the $NNGP$-experiment, run the following bash script.
```
for si in `seq 13 24`; do
    for seed in 0 1 2 3; do
        python nn_inds_to_vecs.py --dim 2 --distro uniform_disk --seed_train $seed --log10N_step 4 --current_step $si &
    wait
    done
done
```

For the $GPnn$-experiments one has to effectively run
```
for si in `seq 12 23`; do
    for seed in 0 1 2 3; do
        python nn_inds_to_vecs.py --dim 4 --distro gaussian --seed_train $seed --log10N_step 2 --current_step $si &
    wait
    done
done
```
and for higher dimensions, replace `--dim 4` with `--dim 8` or `--dim 16`. However, this takes a long time (especially when $d_X=16$ -- up to 48 hours, depending on the CPU), so it is best to run the scripts in parallel (e.g. using array jobs on SLURM).

### 3. Calculate $NNGP/GPnn$ risk vs. training set size.

This part can run on a laptop. To generate the data for $NNGP$, run the following bash script:
```
bash produce_predictions_nngp.sh
```
To generate the data for $GPnn$, run the following bash script:
```
bash produce_predictions_gpnn.sh
```

To produce the plots illustrating Stone's minimax-optimal rates, run the following script:
```
python plots_risk_rates.py --mode NNGP --seeds 0 1 2 3 --nus 0.5 0.75 1.5 2.5
python plots_risk_rates.py --mode GPnn --dims 4 8 16 --seeds 0 1 2 3 --nus 1.0
```

### 4. Calculate $NNGP$ risk vs. hyperparameter landscape.

This part can run on a laptop. Run the following scripts:

```
bash produce_landscape.sh ell
bash produce_landscape.sh sf2
bash produce_landscape.sh sxi2
bash produce_landscape.sh b
```

To produce the plots, run:

```
python plots_risk_landscape.py --seeds 0 1 2 3
```

### 5. Calculate $NNGP$ risk derivatives vs. training set size.

This part can run on a laptop. Run the following script:

```
bash produce_derivatives.sh ell
bash produce_derivatives.sh sf2
bash produce_derivatives.sh sxi2
bash produce_derivatives.sh b
```

And then generate the plots illustrating convergence rates of the risk derivatives:
```
python plots_risk_rates.py --seeds 0 1 2 3
```
