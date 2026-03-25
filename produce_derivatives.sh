#!/usr/bin/env bash

arg="$1"
nu=0.5

for si in `seq 13 24`; do
    if [[ "$arg" == "ell" ]]; then
        par0=2.0
        h=0.01
        n=5

        linspace=($(awk -v par0="$par0" -v h="$h" 'BEGIN {
    		for (i=-2; i<=2; i++) print par0 + i*h
		}'))

        for ((i=0; i<n; i++)); do
            par="${linspace[i]}"
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 0 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose D --sf2_hat 1. --sxi2_hat 0.1 --b1_hat 1. --b2_hat 1. --ell_hat $par --param_ID "$arg" --param_idx $i &
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 1 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose D --sf2_hat 1. --sxi2_hat 0.1 --b1_hat 1. --b2_hat 1. --ell_hat $par --param_ID "$arg" --param_idx $i &
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 2 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose D --sf2_hat 1. --sxi2_hat 0.1 --b1_hat 1. --b2_hat 1. --ell_hat $par --param_ID "$arg" --param_idx $i &
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 3 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose D --sf2_hat 1. --sxi2_hat 0.1 --b1_hat 1. --b2_hat 1. --ell_hat $par --param_ID "$arg" --param_idx $i &
            wait
        done

    elif [[ "$arg" == "sf2" ]]; then
        par0=3.
        h=0.01
        n=5

        linspace=($(awk -v par0="$par0" -v h="$h" 'BEGIN {
    		for (i=-2; i<=2; i++) print par0 + i*h
		}'))

        for ((i=0; i<n; i++)); do
            par="${linspace[i]}"
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 0 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose D --ell_hat 1. --sxi2_hat 0.2 --sxi2 0.2 --b1_hat 1. --b2_hat 1. --sf2_hat $par --param_ID "$arg" --param_idx $i &
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 1 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose D --ell_hat 1. --sxi2_hat 0.2 --sxi2 0.2 --b1_hat 1. --b2_hat 1. --sf2_hat $par --param_ID "$arg" --param_idx $i &
        	python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 2 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose D --ell_hat 1. --sxi2_hat 0.2 --sxi2 0.2 --b1_hat 1. --b2_hat 1. --sf2_hat $par --param_ID "$arg" --param_idx $i &
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 3 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose D --ell_hat 1. --sxi2_hat 0.2 --sxi2 0.2 --b1_hat 1. --b2_hat 1. --sf2_hat $par --param_ID "$arg" --param_idx $i &
            wait
        done

    elif [[ "$arg" == "sxi2" ]]; then
        par0=0.2
        h=0.01
        n=5

        linspace=($(awk -v par0="$par0" -v h="$h" 'BEGIN {
    		for (i=-2; i<=2; i++) print par0 + i*h
		}'))

        for ((i=0; i<n; i++)); do
            par="${linspace[i]}"
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 0 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose D --ell_hat  1. --sf2_hat 1. --b1_hat 1. --b2_hat 1. --sxi2_hat $par --param_ID "$arg" --param_idx $i &
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 1 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose D --ell_hat  1. --sf2_hat 1. --b1_hat 1. --b2_hat 1. --sxi2_hat $par --param_ID "$arg" --param_idx $i &
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 2 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose D --ell_hat  1. --sf2_hat 1. --b1_hat 1. --b2_hat 1. --sxi2_hat $par --param_ID "$arg" --param_idx $i &
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 3 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose D --ell_hat  1. --sf2_hat 1. --b1_hat 1. --b2_hat 1. --sxi2_hat $par --param_ID "$arg" --param_idx $i &
            wait
        done
    elif [[ "$arg" == "b" ]]; then
        par0=0.1
        h=0.01
        n=5

        linspace=($(awk -v par0="$par0" -v h="$h" 'BEGIN {
    		for (i=-2; i<=2; i++) print par0 + i*h
		}'))

        for ((i=0; i<n; i++)); do
            par="${linspace[i]}"
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 0 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose D --ell_hat 1. --sf2_hat 1. --sxi2_hat 0.2 --sxi2 0.2 --b1_hat $par --b2_hat $par --param_ID "$arg" --param_idx $i &
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 1 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose D --ell_hat 1. --sf2_hat 1. --sxi2_hat 0.2 --sxi2 0.2 --b1_hat $par --b2_hat $par --param_ID "$arg" --param_idx $i &
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 2 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose D --ell_hat 1. --sf2_hat 1. --sxi2_hat 0.2 --sxi2 0.2 --b1_hat $par --b2_hat $par --param_ID "$arg" --param_idx $i &
            python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 3 --nu $nu --log10N_step 4 --step_max 24 --current_step $si --purpose D --ell_hat 1. --sf2_hat 1. --sxi2_hat 0.2 --sxi2 0.2 --b1_hat $par --b2_hat $par --param_ID "$arg" --param_idx $i &
            wait
        done
    else
    	echo "Unknown argument: $arg"
    	exit 1
	fi
done