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
python script.py --train-data-size [TRAIN_DATA_SIZE] --val-data-size[VAL_DATA_SIZE] --random-seed [RANDOM_SEED] --representation [REPRESENTATION] --max-epochs [MAX_EPOCHS] --name [EXPERIMENT NAME]
```

Example:
```bash 
python3 script.py --train-data-size 0.9 --val-data-size 0.1 --random-seed 42 --representation n_bins --max-epochs 100 --name n_bins_09^_42
python3 script.py --train-data-size 0.9 --val-data-size 0.1 --random-seed 42 --representation binary --max-epochs 100 --name binary_09_42
```

## New extended script with added representations of time window, spike count and timesurface

Updates:

* Created extended_script.py with new representations of time_window, spike_count and timesurface
* Added the batch size as a new script parameter
* Modified the original model in new_model.py to handle data with different temporal dimensions
    * Instead of inferring the size of the tensors at the initialization of the model, I added an adaptive 3d pooling layer after the final convolution to reshape the data to the same size
    * This won't affect the representations with constant temporal dimensions (n_bins, binary), since I designed the adaptive pooling layer with the idea to have the final features before classification stay the same for these representations, so we don't have to redo the experiments for these
* Added a padding function in /utils for batching the tensors with different temporal dimensions in the dataloaders

Command for updated script:
```bash
python script.py --train-data-size [TRAIN_DATA_SIZE] --val-data-size[VAL_DATA_SIZE] --random-seed [RANDOM_SEED] --representation [REPRESENTATION] --max-epochs [MAX_EPOCHS] --name [EXPERIMENT NAME] --batch-size [BATCH_SIZE]
```