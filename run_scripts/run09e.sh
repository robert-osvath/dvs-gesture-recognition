batch=${1:-8}
repr=${2}
for seed in 42 13 93 45 96 6 98 59 44
do
	for pooling in max avg linear random
	do
		for target_fc in 200 300 400 500 600 700 800
		do
			python3 new_cnn_script.py --train-data-size 0.9 --val-data-size 0.1 --random-seed $seed --conv-layers 3 --representation ${repr} --batch-fc ${target_fc} --max-epochs 100 --name ${repr}_09_${seed}_${batch}_${target_fc} --batch-size ${batch} --output-dir results_with_target_fc
			echo python3 new_cnn_script.py --train-data-size 0.9 --val-data-size 0.1 --random-seed $seed --conv-layers 3 --representation ${repr} --batch-fc ${target_fc} --max-epochs 100 --name ${repr}_09_${seed}_${batch}_${target_fc} --batch-size ${batch} --output-dir results_with_target_fc
		done
	done
done
