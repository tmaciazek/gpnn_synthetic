nu=0.75

for si in `seq 13 24`; do
    python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 0 --nu $nu --log10N_step 4 --step_max 24 --current_step $si &
    wait
done



