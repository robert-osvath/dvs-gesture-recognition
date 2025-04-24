for seed in 42 13 93 45 96 6 98 59 44
do
	for repr in n_bins binary
	do
		#echo python3 extended_script.py --train-data-size 0.9 --val-data-size 0.1 --random-seed $seed --representation ${repr} --max-epochs 100 --name ${repr}_09_${seed}_32 --batch-size 32
		echo python3 script.py --train-data-size 0.9 --val-data-size 0.1 --random-seed $seed --representation ${repr} --max-epochs 100 --name ${repr}_09_${seed}_16 --batch-size 16
	done
done
