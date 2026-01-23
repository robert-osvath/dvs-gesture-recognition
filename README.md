# Experimenting with different representations of event-based data from the DVS Gesture dataset for Gesture Recognition

## Usage

The script trains the base 3-Layer CNN model, as well as runs a test with the test subset of the data, and at the end extracts the user parameters and output metrics in a CSV file. The user willl need to give the following paramters:

* The size of the training data as a percentage of the whole dataset
* The size of the validation data as a percentage of the whole dataset (an early stopping mechanism is implemented based on validation metrics) - train size + val size need to add up to 1
* The random seed
* The representation of the dvs data (the architecture as of now can only handle data with a uniform temporal dimension, so "n_bins" or "binary")
* Maximum epochs for training

Command: 
```bash
python cnn_script.py --train-data-size [TRAIN_DATA_SIZE] --val-data-size[VAL_DATA_SIZE] --random-seed [RANDOM_SEED] --representation [REPRESENTATION] --max-epochs [MAX_EPOCHS] --name [EXPERIMENT NAME]
```

Example:
```bash 
python3 cnn_script.py --train-data-size 0.9 --val-data-size 0.1 --random-seed 42 --representation n_bins --max-epochs 100 --name n_bins_09^_42
python3 cnn_script.py --train-data-size 0.9 --val-data-size 0.1 --random-seed 42 --representation binary --max-epochs 100 --name binary_09_42
```

## Extended script with added representations of time window, spike count and timesurface

Command for updated script:
```bash
python new_cnn_script.py --train-data-size [TRAIN_DATA_SIZE] --val-data-size[VAL_DATA_SIZE] --random-seed [RANDOM_SEED] --representation [REPRESENTATION] --max-epochs [MAX_EPOCHS] --conv-layers [CONV_LAYERS] --name [EXPERIMENT NAME] --batch-size [BATCH_SIZE] --output-dir [OUTPUT_DIR]
```

## Script to train a ResNet model with 

Command:
```bash 
python resnet_script.py --train-data-size [TRAIN_DATA_SIZE] --val-data-size[VAL_DATA_SIZE] --random-seed [RANDOM_SEED] --num-blocks [NUM_RESIDUAL_BLOCKS] --representation [REPRESENTATION] --max-epochs [MAX_EPOCHS] --name [EXPERIMENT NAME] --batch-size [BATCH_SIZE] --patience [PATIENCE] --output-dir [OUTPUT_DIR]
```

## Script to train a 3d Convolutional Spiking Neural Network model

Created a new SNN model that uses 3d convolutions for feature extraction. <br>
**!! You need to be much more careful with the batch sizes when training this model, as the LIF neurons take up a lot of memory. !!**
**Also, you must specify the type of loss function you want the model to optimize, based either on: spike count (count) spike latency (latency), spike frequency (frequency) or max membrane potential (max_mem)**

Command for the snn script:
```bash
python snn_script.py --train-data-size [TRAIN_DATA_SIZE] --val-data-size[VAL_DATA_SIZE] --random-seed [RANDOM_SEED] --representation [REPRESENTATION] --loss [LOSS_FN] --beta [BETA] --max-epochs [MAX_EPOCHS] --name [EXPERIMENT NAME] --batch-size [BATCH_SIZE] --output-dir [OUTPUT_DIR]
```

## Caching the whole dataset

For trying to prevent OOM errors, I've added a script which caches the whole dataset based on a selected representation, which would hopefully make the more complex neural models learn easier.

Command:
```bash
python cache_dataset.py --representation timesurface
```
