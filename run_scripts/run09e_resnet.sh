batch=${1:-8}
repr=${2}
for seed in 42 13 93 45 96 6 98 59 44
do
	for pooling in max avg linear
	do
		for target_fc in 200 300 400 500 600 700 800
		do
			for num_blocks in 1 3 5 7
            do
                res=`cat results_with_target_fc/dvs128_resnet_09_batch8_10seeds.csv | grep ${repr}_09_${seed}_${batch}_${target_fc}_${pooling}_${num_blocks}`
                if [ -z "$res" ]
                then
                    echo Running new script: python3 resnet_script_nocache.py --train-data-size 0.9 --val-data-size 0.1 --random-seed ${seed} --num-blocks ${num_blocks}  --representation ${repr} --batch-fc ${target_fc} --max-epochs 100 --name ${repr}_09_${seed}_${batch}_${target_fc}_${pooling}_${num_blocks} --batch-size ${batch} --pooling-mode ${pooling} --output-dir results_with_target_fc
                    python3 resnet_script_nocache.py --train-data-size 0.9 --val-data-size 0.1 --random-seed ${seed} --num-blocks ${num_blocks} --representation ${repr} --batch-fc ${target_fc} --max-epochs 100 --name ${repr}_09_${seed}_${batch}_${target_fc}_${pooling}_${num_blocks} --batch-size ${batch} --pooling-mode ${pooling} --output-dir results_with_target_fc
                else
                    echo Script already run: python3 resnet_script_nocache.py --train-data-size 0.9 --val-data-size 0.1 --random-seed ${seed} --num-blocks ${num_blocks} --representation ${repr} --batch-fc ${target_fc} --max-epochs 100 --name ${repr}_09_${seed}_${batch}_${target_fc}_${pooling}_${num_blocks} --batch-size ${batch} --pooling-mode ${pooling} --output-dir results_with_target_fc
                fi
              done
		done
	done
done
