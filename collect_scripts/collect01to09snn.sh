batch=${1:-8}
coding=${2:-latency}
count=0
for seed in 42 13 93 #45 96 6 98 59 44
do
	for repr in n_bins binary time_window spike_count timesurface
	do
		for size in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 
		do
	
			res=`find . | grep ${repr}_${coding}_${size}_${seed}_${batch}`
			echo -n "${repr},${coding},${size},0.1,${seed},${batch},0,0,"
			if [ -z "$res" ]
			then
				echo 
				count=$(($count+1))
			else
				cut -f 7- -d "," $res | tail -n 1 | sed "s/tensor(//g" | sed "s/)//g"
			fi	
		done
	done
done
echo \#Missing $count values 
