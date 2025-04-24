for size in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
do
for seed in 42 13 93 
do
	for repr in n_bins binary
	do
		echo python3 script.py --train-data-size ${size} --val-data-size 0.1 --random-seed $seed --representation ${repr} --max-epochs 100 --name ${repr}_09_${seed}_16 --batch-size 16
	done
done
done
