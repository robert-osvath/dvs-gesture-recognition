#!/bin/bash
batch=${1:-8}
for seed in 42 13 93 45 96 6 98 59 44
do
	for repr in n_bins binary time_window spike_count timesurface
	do
		res=`cat results_with_target_fc/dailyaction_cnn_09_batch${batch}_10seeds.csv 2>/dev/null | grep "^${repr}_09_${seed}_${batch},"`
		if [ -z "$res" ]
		then
			echo Running baseline script: python3 new_cnn_script.py --dataset dailyaction --train-data-size 0.9 --val-data-size 0.1 --random-seed $seed --conv-layers 3 --representation ${repr} --max-epochs 100 --name ${repr}_09_${seed}_${batch} --batch-size ${batch} --output-dir results_with_target_fc
			python3 new_cnn_script.py --dataset dailyaction --train-data-size 0.9 --val-data-size 0.1 --random-seed $seed --conv-layers 3 --representation ${repr} --max-epochs 100 --name ${repr}_09_${seed}_${batch} --batch-size ${batch} --output-dir results_with_target_fc
		else
			echo Script already run: python3 new_cnn_script.py --dataset dailyaction --train-data-size 0.9 --val-data-size 0.1 --random-seed $seed --conv-layers 3 --representation ${repr} --max-epochs 100 --name ${repr}_09_${seed}_${batch} --batch-size ${batch} --output-dir results_with_target_fc
		fi
	done
done
