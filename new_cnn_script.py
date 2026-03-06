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
from models.cnn.new_model import Gesture3DConvNet_v2
from dataset.dataset import DVSGestureData
import argparse
import os
import pandas as pd
from utils.pad_tensors import PadTensors, PadTensorsUpdated

device = "gpu" if torch.cuda.is_available() else "cpu"

class GestureRecognition(L.LightningModule):
    def __init__(self, lr, conv_layers, num_classes=11):
        super().__init__()
        self.save_hyperparameters()
        self.model = Gesture3DConvNet_v2(num_classes, conv_layers)
        self.loss = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.confusion_matrix = ConfusionMatrix(
            task="multiclass", num_classes=num_classes
        )

    def training_step(self, batch, batch_idx):
        events, targets = batch
        output = self.model(events)
        loss = self.loss(output["logits"], targets)
        accuracy = self.train_acc(output["probs"], targets)
        self.log("train_loss", loss.item(), on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        events, targets = batch
        output = self.model(events)
        loss = self.loss(output["logits"], targets)
        self.log("val_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)

        accuracy = self.val_acc(output["probs"], targets)
        self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        preds = torch.argmax(output["probs"], dim=1)
        self.confusion_matrix(preds, targets)

        return {"val_loss": loss, "val_acc": accuracy}

    def on_validation_epoch_end(self):
        cm = self.confusion_matrix.compute().cpu().numpy()
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted Labels")
        ax.set_ylabel("True Labels")
        ax.set_title("Confusion Matrix")

        writer = SummaryWriter(log_dir=self.logger.log_dir)
        writer.add_figure("Confusion Matrix", fig, global_step=self.current_epoch)
        writer.close()

        self.confusion_matrix.reset()

    def test_step(self, batch, batch_idx):
        events, targets = batch
        output = self.model(events)
        loss = self.loss(output["logits"], targets)

        accuracy = self.test_acc(output["probs"], targets)

        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

        return {"test_loss": loss, "test_acc": accuracy}

    def predict_step(self, batch, batch_idx):
        events, _ = batch
        output = self.model(events)
        return torch.argmax(output["probs"], dim=1)

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
        default_root_dir="../checkpoints",
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
        "--conv-layers", type=int, required=True, help="Set the number of convolutional layers."
    )

    parser.add_argument(
        "--representation",
        type=str,
        choices=["n_bins", "binary", "time_window", "spike_count", "timesurface"],
        required=True,
        help="The event representation of the DVS data to use.",
    )

    parser.add_argument(
        "--batch-fc", type=str, help="Desired frame count of the batches."
    )

    parser.add_argument(
        "--pooling-mode", type=str, default="max", help="Set the pooling mode."
    )

    parser.add_argument(
        "--max-epochs", type=int, required=True, help="Set the max epochs."
    )

    parser.add_argument(
        "--batch-size", type=int, default=10, help="Set the batch size."
    )

    parser.add_argument(
        "--name", type=str, required=True, help="Set the experiment name."
    )
    
    parser.add_argument(
        "--output-dir", type=str, required=True, help="Set the output dir name."
    )

    args = parser.parse_args()
    train_data_size = args.train_data_size
    val_data_size = args.val_data_size
    random_seed = args.random_seed
    conv_layers = args.conv_layers
    representation = args.representation
    desired_frame_count = args.batch_fc
    pooling_mode = args.pooling_mode
    max_epochs = args.max_epochs
    exp_name = args.name
    batch_size = args.batch_size
    output_dir = args.output_dir

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
            dt = 10000
        )
    else:
        raise ValueError("Invalid representation.")

    # Validate train and val data size
    if train_data_size + val_data_size > 1 or train_data_size < 0 or val_data_size < 0:
        raise ValueError("Invalid train or val data size.")

    # Load the datasets
    print("splitting with ",train_data_size,val_data_size,1-train_data_size-val_data_size)

    train_data, val_data, _ = random_split(
        tonic.datasets.DVSGesture(save_to=("./data"), train=True, transform=transform),
        [train_data_size, val_data_size, round((1 - train_data_size - val_data_size)*100)/100],
    )

    test_data = tonic.datasets.DVSGesture(
        save_to=("./data"), train=False, transform=transform
    )

    # Create the transformed datasets
    train_dataset = DVSGestureData(train_data)
    val_dataset = DVSGestureData(val_data)
    test_dataset = DVSGestureData(test_data)

    # Create the data loaders
    # !! Make sure to change num_workers accordingly !!
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    # val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    collate_fn = PadTensorsUpdated(target_frames=int(desired_frame_count), mode=pooling_mode)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=8)

    # Create the model
    model = GestureRecognition(1e-3, conv_layers)

    # Alternatively, you can load the model from a checkpoint
    # model = GestureRecognition.load_from_checkpoint("checkpoints/a.ckpt")

    # Create the trainer
    trainer = train_setup(max_epochs=max_epochs, patience=2, log_dir="./logs")

    # Train the model
    train(model, train_loader, val_loader, trainer)

    train_metrics = trainer.callback_metrics.copy()
    train_epochs = trainer.current_epoch

    # Test the model
    test(model, test_loader, trainer)

    test_metrics = trainer.callback_metrics.copy()

    # Get the number of epochs used for training
    num_epochs = trainer.current_epoch + 1

    # Save script params and outputs in a csv file
    df = pd.DataFrame(
        {
            "name": [exp_name],
            "train_data_size": [train_data_size],
            "val_data_size": [val_data_size],
            "random_seed": [random_seed],
            "representation": [representation],
            "train_acc": [train_metrics["train_acc"].item()],
            "val_acc": [train_metrics["val_acc"].item()],
            "test_acc": [test_metrics["test_acc"].item()],
            "num_epochs": [train_epochs],
            "batch_size": [batch_size],
            "target_frames": [desired_frame_count],
            "pooling_mode": [pooling_mode]
        }
    )
    filename = "%s/dvs128_cnn_09_batch%s_10seeds.csv"%(output_dir, batch_size)
    file_exists = os.path.isfile(filename)
    #df.to_csv(filename, mode="a", header=not file_exists, index=False)
    df.to_csv(filename, mode="a", header=not file_exists, index=False)



if __name__ == "__main__":
    main()
