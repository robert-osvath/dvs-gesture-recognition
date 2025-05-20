batch=${1:-8}
for seed in 42 13 93 45 96 6 98 59 44
do
	for repr in n_bins binary time_window spike_count #timesurface
	do
		for coding in latency frequency
		do
			echo python3 snn_script.py --loss ${coding} --train-data-size 0.9 --val-data-size 0.1 --random-seed $seed --representation ${repr} --max-epochs 100 --name ${repr}_${coding}_09_${seed}_${batch} --batch-size ${batch}
		done
	done
done
