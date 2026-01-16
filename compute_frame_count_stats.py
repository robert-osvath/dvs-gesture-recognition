import tonic
import tonic.transforms as TT
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import os
from models.resnet.resnet import ResNet3D
from dataset.dataset import DVSGestureData, CachedDVSGestureData
import argparse
import os
import pandas as pd
from utils.pad_tensors import FrameCountStats

def count_frames(args):
    # Data transforms
    output_dir=args.output_dir
    representation=args.representation
    exp_name=representation

    sensor_size = tonic.datasets.DVSGesture.sensor_size
    if representation == "n_bins":
        transform = TT.ToFrame(sensor_size=sensor_size, n_time_bins=100)
    elif representation == "binary":
        transform = TT.Compose(
            [
                TT.ToFrame(sensor_size=sensor_size, n_time_bins=100 * 2),
                TT.ToBinaRep(n_frames=100, n_bits=2),
            ]
        )
    elif representation == "time_window":
        transform = TT.ToFrame(sensor_size=sensor_size, time_window=10000)
    elif representation == "spike_count":
        transform = TT.ToFrame(sensor_size=sensor_size, event_count=1000)
    elif representation == "timesurface":
        transform = TT.ToTimesurface(
            sensor_size=sensor_size, 
            tau=30000,
            dt=10000
        )
    else:
        raise ValueError("Invalid representation.")

    train_dataset = DVSGestureData(tonic.datasets.DVSGesture(save_to=("./data"), train=True, transform=transform))

    test_dataset = DVSGestureData(tonic.datasets.DVSGesture(save_to=("./data"), train=False, transform=transform))


    print(train_dataset)

    collate_fn=FrameCountStats(batch_first=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,num_workers=args.num_workers)
    
    print(train_loader)
    for i,(x,y) in enumerate(train_loader):
        print("processing batch %d running_mean ",collate_fn.getRunningMeanFrameCount())
        break
        pass

    collate_fn=FrameCountStats(batch_first=True,
            min=collate_fn.getMinFrameCount(),
            max=collate_fn.getMaxFrameCount(),
            running_mean=collate_fn.getRunningMeanFrameCount(),
            histo=collate_fn.getHistoFrameCount()
            )
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,num_workers=args.num_workers)
    for x,y in test_loader:
        pass

    # Save script params and outputs in a csv file
    df = pd.DataFrame(
        {
            "representation": [representation],
            "max_frame_count": [collate_fn.getMaxFrameCount()],
            "min_frame_count": [collate_fn.getMinFrameCount()],
            "running_mean_frame_count": [collate_fn.getRunningMeanFrameCount()],
        }
    )
    df = pd.concat(df,pd.DataFrame.fromDict(collate_fn.getHisto()))

    filename = "%s/%s_counts.csv"%(output_dir,exp_name)
    file_exists = os.path.isfile(filename)

    print ("Saving data to %s"%filename)
    #df.to_csv(filename, mode="a", header=not file_exists, index=False)

    df.to_csv(filename, mode="w", header=True, index=False)
def main():
    parser = argparse.ArgumentParser(
        description="A script that takes command-line arguments."
    )

    parser.add_argument(
        "--output-dir", type=str, required=True, help="Set the output dir name."
    )
    
    parser.add_argument(
        "--batch-size", type=int, default=256, help="Set the batch size."
    )
    
    parser.add_argument(
        "--num-workers", type=int, default=48, help="Set the number of workers for data loader."
    )
    
    representations=["n_bins", "binary", "time_window", "spike_count", "timesurface"]

    args = parser.parse_args()
    output_dir = args.output_dir
    print("Preparing to save results to %s"%output_dir)
    for representation in representations:
        print("counting %s"%representation)
        args.representation=representation
        count_frames(args)


if __name__ == "__main__":
    main()
