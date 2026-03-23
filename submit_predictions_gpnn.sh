nu=1.0

for si in `seq 12 23`; do
    python predictions.py --mode GPnn --distro gaussian --dim 16 --seed_train 0 --nu $nu --log10N_step 2 --step_max 23 --current_step $si --sxi2 0.2 --sf2_hat 1. --sxi2_hat 0.15 --ell_hat 0.5 &
    wait
done



