import tonic
import tonic.transforms as TT
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, ConfusionMatrix
import lightning.pytorch as L
from torch.utils.data import random_split
import os
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from snn_model import Gesture3DCSNN
from dataset import DVSGestureData
import argparse
import os
import pandas as pd
from utils.pad_tensors import PadTensors
import snntorch as snn
from snntorch import surrogate, functional as SF
from tonic import DiskCachedDataset


#device = "gpu" if torch.cuda.is_available() else "cpu"
device = "cuda" if torch.cuda.is_available() else "cpu"

class GestureRecognition(L.LightningModule):
  def __init__(self, lr, loss_fn, beta, num_classes=11):
    super().__init__()
    self.save_hyperparameters()
    self.model = Gesture3DCSNN(beta, num_classes).to(device)
    # self.loss_fn = SF.ce_count_loss()
    # self.loss_fn = SF.ce_temporal_loss()
    self.loss_fn = loss_fn
    self.confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)

  def training_step(self, batch, batch_idx):
    events, targets = batch
    events, targets = events.to(device).float(), targets.to(device)

    output = self.model(events)
    spike_rec = output['spike_rec']
    mem_rec = output['mem_rec']
    probs = output['probs']

    print(f"Output dtype: {spike_rec.dtype}")  # Should be float32
    print(f"Targets dtype: {targets.dtype}")         # Should be int64 (long)
    # loss = self.loss_fn(spike_rec, targets)
    if isinstance(self.loss_fn, SF.ce_max_mem_loss):
       loss = self.loss_fn(mem_rec, targets)
    else:
       loss = self.loss_fn(spike_rec, targets)
    self.log('train_loss', loss.item(), on_step=True, on_epoch=True, prog_bar=True)

    return loss

  def validation_step(self, batch, batch_idx):
    events, targets = batch
    events, targets = events.to(device).float(), targets.to(device)

    output = self.model(events)
    spike_rec = output['spike_rec']
    probs = output['probs']

    loss = self.loss_fn(spike_rec, targets)
    self.log('val_loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True)

    rate_acc = SF.accuracy_rate(spike_rec, targets)
    self.log('val_rate_acc', rate_acc, on_step=False, on_epoch=True, prog_bar=True)

    temporal_acc = SF.accuracy_temporal(spike_rec, targets)
    self.log('val_temp_acc', temporal_acc, on_step=False, on_epoch=True, prog_bar=True)

    self.confusion_matrix(probs, targets)

    return {'val_loss': loss, 'val_rate_acc': rate_acc, 'val_temp_acc': temporal_acc}

  def on_validation_epoch_end(self):
    cm = self.confusion_matrix.compute().cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')

    writer = SummaryWriter(log_dir=self.logger.log_dir)
    writer.add_figure('Confusion Matrix', fig, global_step=self.current_epoch)
    writer.close()

    self.confusion_matrix.reset()

  def test_step(self, batch, batch_idx):
    events, targets = batch
    events, targets = events.to(device).float(), targets.to(device)

    output = self.model(events)
    spike_rec = output['spike_rec']
    probs = output['probs']

    rate_acc = SF.accuracy_rate(spike_rec, targets)
    self.log('test_rate_acc', rate_acc, on_step=False, on_epoch=True, prog_bar=True)

    temporal_acc = SF.accuracy_temporal(spike_rec, targets)
    self.log('test_temp_acc', temporal_acc, on_step=False, on_epoch=True, prog_bar=True)

    return {'test_rate_acc': rate_acc, 'test_temp_acc': temporal_acc}

  def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=self.hparams.lr)
  

def train_setup(max_epochs, patience, log_dir):
    # Set up the early stopping callback
    early_stopping = EarlyStopping(
        monitor="val_loss", patience=patience, mode="min", verbose=True
    )

    # Set up the trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=2,
        accelerator=device,
        devices="auto",
        precision="16-mixed",
        logger=L.loggers.TensorBoardLogger(log_dir, name="gesture_recognition"),
        log_every_n_steps=1,
        callbacks=[early_stopping],
        default_root_dir="./checkpoints",
    )

    return trainer


def train(model, train_loader, val_loader, trainer):
    trainer.fit(model, train_loader, val_loader)


def test(model, test_loader, trainer):
    trainer.test(model, test_loader)

def main():
    parser = argparse.ArgumentParser(
        description="A script that takes command-line arguments."
    )
    parser.add_argument(
        "--train-data-size",
        type=float,
        required=True,
        help="The size of the data to use for training (as a fraction of the whole dataset).",
    )

    parser.add_argument(
        "--val-data-size",
        type=float,
        required=True,
        help="The size of the data to use for validation (as a fraction of the whole dataset).",
    )

    parser.add_argument(
        "--random-seed", type=int, required=True, help="Set the random seed."
    )

    parser.add_argument(
        "--representation",
        type=str,
        choices=["n_bins", "binary", "time_window", "spike_count", "timesurface"],
        required=True,
        help="The event representation of the DVS data to use.",
    )

    parser.add_argument(
       "--loss",
       type=str,
       choices=["latency", "frequency", "count", "max_mem"],
       required=True,
       help="The type of loss function the model should optimize: based either on spike count (count), spike timing (latency), spike rate (frequency), or max membrane potential (max_mem)"
    )

    parser.add_argument(
        "--beta", type=float, required=True, help="Set the beta value."
    )

    parser.add_argument(
        "--max-epochs", type=int, required=True, help="Set the max epochs."
    )

    parser.add_argument(
        "--batch-size", type=int, required=True, help="Set the batch size."
    )

    parser.add_argument(
        "--name", type=str, required=True, help="Set the experiment name."
    )

    args = parser.parse_args()
    train_data_size = args.train_data_size
    val_data_size = args.val_data_size
    random_seed = args.random_seed
    representation = args.representation
    loss = args.loss
    max_epochs = args.max_epochs
    beta = args.beta
    exp_name = args.name
    batch_size = args.batch_size

    # Set the random seed
    L.seed_everything(random_seed, workers=True)

    # Data transforms
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
    
    if loss == "latency":
       loss_fn = SF.ce_temporal_loss()
    elif loss == "frequency":
       loss_fn = SF.ce_rate_loss()
    elif loss == "count":
       loss_fn = SF.ce_count_loss()
    elif loss == "max_mem":
       loss_fn = SF.ce_max_membrane_loss()
    else:
       raise ValueError("Invalid loss function.")

    # Validate train and val data size
    if train_data_size + val_data_size > 1+1e4 or train_data_size < 0 or val_data_size < 0:
        raise ValueError("Invalid train or val data size.")

    # Load the datasets
    left_data=1-train_data_size-val_data_size
    left_data=left_data if left_data > 0 else 0
    print("splitting with ",train_data_size,val_data_size,left_data)

    train_data, val_data, _ = random_split(
        tonic.datasets.DVSGesture(save_to=("./data"), train=True, transform=transform),
        [train_data_size, val_data_size, left_data],
    )

    test_data = tonic.datasets.DVSGesture(
        save_to=("./data"), train=False, transform=transform
    )


    # Create the data loaders
    # !! Make sure to change num_workers accordingly !!
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    cached_train_set = DiskCachedDataset(train_data, cache_path='./cache/train')
    cached_val_set = DiskCachedDataset(val_data, cache_path='./cache/val')

    train_loader = DataLoader(cached_train_set, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=True), shuffle=True)
    val_loader = DataLoader(cached_val_set, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=True), shuffle=False)
    test_loader = DataLoader(test_data, batch_size=batch_size, collate_fn=tonic.collation.PadTensors(batch_first=True), shuffle=False)

    # Create the model
    model = GestureRecognition(lr=0.001, loss_fn=loss_fn, beta=beta)

    # Alternatively, you can load the model from a checkpoint
    # model = GestureRecognition.load_from_checkpoint("checkpoints/a.ckpt")

    # Create the trainer
    trainer = train_setup(max_epochs=max_epochs, patience=2, log_dir="./logs")

    # Train the model
    train(model, train_loader, val_loader, trainer)

    # Test the model
    test(model, test_loader, trainer)

    # Get the number of epochs used for training
    num_epochs = trainer.current_epoch + 1

    # Save script params and outputs in a csv file
    df = pd.DataFrame(
        {
            "train_data_size": [train_data_size],
            "val_data_size": [val_data_size],
            "random_seed": [random_seed],
            "representation": [representation],
            "val_rate_acc": [trainer.callback_metrics.get('val_rate_acc', 0.0)],
            "val_temp_acc": [trainer.callback_metrics.get('val_temp_acc', 0.0)],
            "test_rate_acc": [trainer.callback_metrics.get('test_rate_acc', 0.0)],
            "test_temp_acc": [trainer.callback_metrics.get('test_temp_acc', 0.0)],
            "num_epochs": [num_epochs],
            "batch_size": [batch_size],
            "loss": [loss],
            "beta": [beta]
        }
    )
    filename = "%s_params_and_outputs.csv"%(exp_name)
    file_exists = os.path.isfile(filename)
    df.to_csv(filename, mode="a", header=not file_exists, index=False)


if __name__ == "__main__":
    main()
