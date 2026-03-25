#!/usr/bin/env bash

arg="$1"
nu=0.5

for si in 16 20 24; do
    if [[ "$arg" == "ell" ]]; then
        start=0.5
        end=2.5
        n=40

        linspace=($(awk -v a="$start" -v b="$end" -v n="$n" '
        BEGIN {
            for (i = 0; i < n; i++) {
                print a + i*(b-a)/(n-1)
            }
        }'))

        for ((i=0; i<n; i++)); do
            par="${linspace[i]}"
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 0 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose landscape --ell_hat $par --param_ID "$arg" --param_idx $i &
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 1 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose landscape --ell_hat $par --param_ID "$arg" --param_idx $i &
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 2 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose landscape --ell_hat $par --param_ID "$arg" --param_idx $i &
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 3 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose landscape --ell_hat $par --param_ID "$arg" --param_idx $i &
            wait
        done

    elif [[ "$arg" == "sf2" ]]; then
        start=1.
        end=4.
        n=40

        linspace=($(awk -v a="$start" -v b="$end" -v n="$n" '
        BEGIN {
            for (i = 0; i < n; i++) {
                print a + i*(b-a)/(n-1)
            }
        }'))

        for ((i=0; i<n; i++)); do
            par="${linspace[i]}"
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 0 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose landscape --sf2_hat $par --param_ID "$arg" --param_idx $i &
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 1 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose landscape --sf2_hat $par --param_ID "$arg" --param_idx $i &
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 2 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose landscape --sf2_hat $par --param_ID "$arg" --param_idx $i &
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 3 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose landscape --sf2_hat $par --param_ID "$arg" --param_idx $i &
            wait
        done

    elif [[ "$arg" == "sxi2" ]]; then
        start=0.05
        end=0.3
        n=40

        linspace=($(awk -v a="$start" -v b="$end" -v n="$n" '
        BEGIN {
            for (i = 0; i < n; i++) {
                print a + i*(b-a)/(n-1)
            }
        }'))

        for ((i=0; i<n; i++)); do
            par="${linspace[i]}"
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 0 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose landscape --sxi2_hat $par --param_ID "$arg" --param_idx $i &
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 1 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose landscape --sxi2_hat $par --param_ID "$arg" --param_idx $i &
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 2 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose landscape --sxi2_hat $par --param_ID "$arg" --param_idx $i &
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 3 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose landscape --sxi2_hat $par --param_ID "$arg" --param_idx $i &
            wait
        done
    elif [[ "$arg" == "b" ]]; then
        start=0.0
        end=2.0
        n=40

        linspace=($(awk -v a="$start" -v b="$end" -v n="$n" '
        BEGIN {
            for (i = 0; i < n; i++) {
                print a + i*(b-a)/(n-1)
            }
        }'))

        for ((i=0; i<n; i++)); do
            par="${linspace[i]}"
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 0 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose landscape --b1_hat $par --b2_hat $par --param_ID "$arg" --param_idx $i &
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 1 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose landscape --b1_hat $par --b2_hat $par --param_ID "$arg" --param_idx $i &
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 2 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose landscape --b1_hat $par --b2_hat $par --param_ID "$arg" --param_idx $i &
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 3 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose landscape --b1_hat $par --b2_hat $par --param_ID "$arg" --param_idx $i &
            wait
        done
    else
    	echo "Unknown argument: $arg"
    	exit 1
	fi
done