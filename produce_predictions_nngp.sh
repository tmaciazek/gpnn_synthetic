for nu in 0.5 0.75 1.5 2.5; do
	for si in `seq 13 24`; do
    	python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 0 --nu $nu --log10N_step 4 --step_max 24 --current_step $si &
    	python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 1 --nu $nu --log10N_step 4 --step_max 24 --current_step $si &
    	python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 2 --nu $nu --log10N_step 4 --step_max 24 --current_step $si &
    	python predictions.py --mode NNGP --distro uniform_disk --dim 2 --seed_train 3 --nu $nu --log10N_step 4 --step_max 24 --current_step $si &
    	wait
	done
done



