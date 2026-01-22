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
from models.resnet.resnet import ResNet3D
from dataset.dataset import DVSGestureData, CachedDVSGestureData
import argparse
import os
import pandas as pd
from utils.pad_tensors import PadTensors, PadTensorsUpdated

device = "gpu" if torch.cuda.is_available() else "cpu"

print ("Running on device %s"%device)


class ResNet3DModule(L.LightningModule):
    def __init__(self, num_blocks, lr=1e-3, num_classes=11):
        super().__init__()
        self.model = ResNet3D(num_blocks, num_classes)
        self.loss = nn.CrossEntropyLoss()
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.confusion_matrix = ConfusionMatrix(task="multiclass", num_classes=num_classes)
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        events, target = batch
        output = self.model(events)
        loss = self.loss(output["logits"], target)
        self.train_acc(output["probs"], target)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        events, target = batch
        output = self.model(events)
        loss = self.loss(output["logits"], target)
        self.val_acc(output["probs"], target)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True, prog_bar=True)


        preds = torch.argmax(output["probs"], dim=1)
        self.confusion_matrix(preds, target)
        return loss

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
        events, target = batch
        output = self.model(events)
        loss = self.loss(output["logits"], target)
        self.test_acc(output["probs"], target)
        self.log('test_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('test_acc', self.test_acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

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
        "--num-blocks", type=int, required=True, help="Set the number of residual blocks."
    )

    parser.add_argument(
        "--representation",
        type=str,
        choices=["n_bins", "binary", "time_window", "spike_count", "timesurface"],
        required=True,
        help="The event representation of the DVS data to use.",
    )

    parser.add_argument(
        "--max-epochs", type=int, required=True, help="Set the max epochs."
    )

    parser.add_argument(
        "--patience", type=int, required=False, help="Set the early stopping patience.", default=2
    )

    parser.add_argument(
        "--batch-size", type=int, default=10, help="Set the batch size."
    )
    
    parser.add_argument(
        "--desired-fc", type=int, default=100, help="Set desired frame count per sample."
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
    num_blocks = args.num_blocks
    representation = args.representation
    max_epochs = args.max_epochs
    patience = args.patience
    exp_name = args.name
    batch_size = args.batch_size
    output_dir = args.output_dir

    print("Preparing to save results to %s"%output_dir)

    # Set the random seed
    L.seed_everything(random_seed, workers=True)

    # Data transforms -- not needed for cached data
    # sensor_size = tonic.datasets.DVSGesture.sensor_size
    # if representation == "n_bins":
    #     transform = TT.ToFrame(sensor_size=sensor_size, n_time_bins=100)
    # elif representation == "binary":
    #     transform = TT.Compose(
    #         [
    #             TT.ToFrame(sensor_size=sensor_size, n_time_bins=100 * 2),
    #             TT.ToBinaRep(n_frames=100, n_bits=2),
    #         ]
    #     )
    # elif representation == "time_window":
    #     transform = TT.ToFrame(sensor_size=sensor_size, time_window=10000)
    # elif representation == "spike_count":
    #     transform = TT.ToFrame(sensor_size=sensor_size, event_count=1000)
    # elif representation == "timesurface":
    #     transform = TT.ToTimesurface(
    #         sensor_size=sensor_size, 
    #         tau=30000,
    #         dt=10000
    #     )
    # else:
    #     raise ValueError("Invalid representation.")

    # Validate train and val data size
    if train_data_size + val_data_size > 1 or train_data_size < 0 or val_data_size < 0:
        raise ValueError("Invalid train or val data size.")
    
    cached_train_dataset = CachedDVSGestureData(cache_path=os.path.join('./cache', representation, 'train'))
    cached_test_dataset = CachedDVSGestureData(cache_path=os.path.join('./cache', representation, 'test'))

    # Load the datasets
    print("splitting with ",train_data_size,val_data_size,1-train_data_size-val_data_size)

    train_data, val_data, _ = random_split(
        cached_train_dataset,
        [train_data_size, val_data_size, round((1 - train_data_size - val_data_size)*100)/100],
    )

    test_data = cached_test_dataset

    collate_fn=PadTensors(batch_first=True)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    #Create the model
    model = ResNet3DModule(num_blocks=num_blocks, lr=1e-3, num_classes=11)

    # Create the trainer
    trainer = train_setup(max_epochs=max_epochs, patience=patience, log_dir="./logs")

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
            "train_acc": [trainer.callback_metrics["train_acc"].item()],
            "val_acc": [trainer.callback_metrics["val_acc"].item()],
            "test_acc": [trainer.callback_metrics["test_acc"].item()],
            "num_epochs": [num_epochs],
            "batch_size": [batch_size],
            "num_blocks": [num_blocks],
        }
    )
    filename = "%s/%s_params_and_outputs.csv"%(output_dir,exp_name)
    file_exists = os.path.isfile(filename)

    print ("Saving data to %s"%filename)
    #df.to_csv(filename, mode="a", header=not file_exists, index=False)

    df.to_csv(filename, mode="w", header=True, index=False)


if __name__ == "__main__":
    main()
