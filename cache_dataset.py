import os
import torch
import tonic
import tonic.transforms as TT
import argparse
from tqdm import tqdm

def cache_all_data(train_set, test_set, representation, path='./cache'):    
    for idx, (event, target) in enumerate(tqdm(train_set, desc="Caching train set")):
        cached_path = os.path.join(path, representation, 'train', f"sample_{idx}.pt")
        os.makedirs(os.path.dirname(cached_path), exist_ok=True)
        torch.save((event, target), cached_path)
   
    for idx, (event, target) in enumerate(tqdm(test_set, desc="Caching test set")):
        cached_path = os.path.join(path, representation, 'test', f"sample_{idx}.pt")
        os.makedirs(os.path.dirname(cached_path), exist_ok=True)
        torch.save((event, target), cached_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A script that takes command-line arguments."
    )
    parser.add_argument(
        "--representation",
        type=str,
        choices=["n_bins", "binary", "time_window", "spike_count", "timesurface"],
        required=True,
        help="The event representation of the DVS data to use.",
    )

    args = parser.parse_args()
    representation = args.representation
    
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

    train_set = tonic.datasets.DVSGesture(save_to='./data', train=True, transform=transform)
    test_set = tonic.datasets.DVSGesture(save_to='./data', train=False, transform=transform)
    cache_all_data(train_set, test_set, representation)