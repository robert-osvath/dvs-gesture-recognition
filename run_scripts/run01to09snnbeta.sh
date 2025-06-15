batch=${1:-8}
coding=${2:-latency}
output_dir=${3:-.}
for seed in 42 13 93 #45 96 6 98 59 44
do
	for repr in n_bins binary time_window spike_count timesurface
	do
		#for size in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 
		for size in 0.1 0.3 0.5 0.7 0.9 
		do
			for beta in 0.9
			do
				echo python3 snn_script.py --loss ${coding} --train-data-size ${size} --val-data-size 0.1 --random-seed $seed --representation ${repr} --max-epochs 100 --name ${repr}_${coding}_${size}_${seed}_${batch}_${beta} --batch-size ${batch} --beta ${beta} --output-dir ${output_dir}
			done
		done
	done
done
