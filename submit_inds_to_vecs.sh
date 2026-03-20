for p in `seq 13 24`
do
    python nn_inds_to_vecs.py --dim 2 --distro uniform_disk --seed_train 0 --log10_tot_size 6 --out_prefix exactNN_uniform_disk_d2_k100_seed0_N1e${p}div4 &
    wait
done


