for seed in 42 13 93 45 96 6 98 59 44
do
	for repr in time_window spike_count timesurface 
	do
		echo python3 extended_script.py --train-data-size 0.9 --val-data-size 0.1 --random-seed $seed --representation ${repr} --max-epochs 100 --name ${repr}_09_${seed}_8 --batch-size 8 
	done
done
