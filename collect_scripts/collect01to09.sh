batch=${1:-8}
folder=${2:-.}
count=0
echo "representation,train_size,val_size,seed,bach_size,,,train_acc,val_acc,test_acc,epoch,loss=representation,beta"
for seed in 6 #42 13 93 #45 96 6 98 59 44
do
	for repr in n_bins binary time_window spike_count timesurface
	do
		#for size in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 
		for size in 0.1 0.3 0.5 0.7 0.9 
		do
	
			res=`find ${folder} | grep ${repr}_${size}_${seed}_${batch}`
			echo -n "${repr},${size},0.1,${seed},${batch},0,0,"
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
