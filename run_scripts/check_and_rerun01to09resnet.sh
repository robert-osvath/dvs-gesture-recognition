batch=${1:-8}
folder=${2:-.}
count=0
for seed in 42 13 93 #45 96 6 98 59 44
do
	for repr in n_bins binary time_window spike_count timesurface
	do
	    for blocks in 1 3 5 10
	    do
		#for size in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 
		for size in 0.1 0.4 0.5 0.7 0.9 
		do
			res=`find ${folder} | grep ${repr}_${size}_${blocks}_${seed}_${batch}`
			if [ -z "$res" ]
			then
				echo \#Missing ${repr}_${size}_${blocks}_${seed}_${batch} 
				echo python3 resnet_script.py --train-data-size ${size} --val-data-size 0.1 --random-seed $seed --num-blocks ${blocks} --representation ${repr} --max-epochs 100 --name ${repr}_${size}_${blocks}_${seed}_${batch} --batch-size ${batch} --output-dir=${folder}
				count=$(($count+1))
			else
				echo \#OK ${repr}_${size}_${blocks}_${seed}_${batch} 
			fi	
		done
	    done
	done
done
echo \#Regenerated $count scripts
