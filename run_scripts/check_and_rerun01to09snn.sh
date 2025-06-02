batch=${1:-8}
coding=${2:-latency}
count=0
for seed in 42 13 93 #45 96 6 98 59 44
do
	for repr in n_bins binary time_window spike_count timesurface
	do
		for size in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 
		do
			res=`ls -1 * | grep ${repr}_${coding}_${size}_${seed}_${batch}`
			if [ -z $res ]
			then
				echo \#Missing ${repr}_${coding}_${size}_${seed}_${batch} 
				echo python3 snn_script.py --loss ${coding} --train-data-size ${size} --val-data-size 0.1 --random-seed $seed --representation ${repr} --max-epochs 100 --name ${repr}_${coding}_${size}_${seed}_${batch} --batch-size ${batch} --beta 0.5
				count=$(($count+1))
			else
				echo \#OK ${repr}_${coding}_${size}_${seed}_${batch} 
			fi	
		done
	done
done
echo \#Regenerated $count scripts
