for si in `seq 13 24`
do
    python nn_inds_to_vecs.py --dim 2 --distro uniform_disk --seed_train 3 --log10N_step 4 --current_step $si &
    wait
done


