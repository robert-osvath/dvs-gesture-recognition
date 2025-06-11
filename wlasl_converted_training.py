import tonic
import tonic.transforms as TT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.v2 as T
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from torchmetrics import Accuracy, ConfusionMatrix
import lightning.pytorch as L
from torch.utils.data import random_split
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
import json
import os
from utils.pad_tensors import PadTensors
import argparse

sensor_size = (128, 128, 2)
transform = TT.ToFrame(sensor_size=sensor_size, time_window=10000)

device = "gpu" if torch.cuda.is_available() else "cpu"

class NWLASLDataset(Dataset):
  def __init__(self, base_path, transform=None):
    self.transform = transform
    self._base_path = base_path
    self.videos = []
    for target_dir in os.listdir(self._base_path):
      target_dir_path = os.path.join(self._base_path, target_dir)
      if os.path.isdir(target_dir_path):
        for video_dir in os.listdir(target_dir_path):
          video_dir_path = os.path.join(target_dir_path, video_dir)
          if os.path.isdir(video_dir_path):
            for video_file in os.listdir(video_dir_path):
              if video_file.endswith('.aedat4'):
                video_path = os.path.join(video_dir_path, video_file)
                self.videos.append({"aedat_path": video_path, "target": int(target_dir) - 1})

  def __len__(self):
    return len(self.videos)

  def __getitem__(self, idx):
    video_path = self.videos[idx]["aedat_path"]
    try:
      events = tonic.io.read_aedat4(video_path)
    except RuntimeError as e:
      print(f"Error reading file: {video_path}")
      print(f"Error message: {e}")
      raise RuntimeError(f"Error reading file: {video_path}")

    if self.transform:
      events = self.transform(events)

    if events.ndim == 4:
        events = torch.tensor(events.transpose(1, 0, 2, 3), dtype=torch.float32)
    elif events.ndim == 3:
        events = torch.tensor(events.transpose(0, 1, 2).unsqueeze(1), dtype=torch.float32) # (T, C=1, H, W)
    else:
        print(f"Unexpected events shape after transform: {events.shape} for file: {video_path}")
        raise ValueError(f"Unexpected events shape after transform: {events.shape}")


    target = self.videos[idx]["target"]
    target = torch.tensor(target, dtype=torch.long)
    return events, target
  
class ASL3DConvNet(nn.Module):
  def __init__(self, classes):
    super(ASL3DConvNet, self).__init__()
    self.conv1 = self._make_conv_layer(2, 8)
    self.conv2 = self._make_conv_layer(8, 16)
    self.conv3 = self._make_conv_layer(16, 32)

    self.adaptive_pool = nn.AdaptiveAvgPool3d((6, 8, 8))

    self.fc1 = nn.Linear(32 * 6 * 8 * 8, 2048)
    self.fc2 = nn.Linear(2048, classes)

  def _make_conv_layer(self, in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, stride=2)
    )

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.adaptive_pool(x)
    x = x.view(x.size(0), -1)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return {
        "logits": x,
        "probs": F.softmax(x, dim=1)
    }
  
class SignLanguageRecognition(L.LightningModule):
  def __init__(self, lr, num_classes):
    super().__init__()
    self.save_hyperparameters()
    self.model = ASL3DConvNet(num_classes).to(device)
    self.loss = nn.CrossEntropyLoss()
    self.acc = Accuracy(task="multiclass", num_classes=num_classes)

  def training_step(self, batch, batch_idx):
    events, targets = batch
    events = events.to(device)
    targets = targets.to(device)

    output = self.model(events)
    loss = self.loss(output["logits"], targets)
    self.log("train_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)

    return loss

  def validation_step(self, batch, batch_idx):
    events, targets = batch
    events = events.to(device)
    targets = targets.to(device)

    output = self.model(events)
    loss = self.loss(output["logits"], targets)
    self.log("val_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True)

    accuracy = self.acc(output["probs"], targets)
    self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

    return {"val_loss": loss, "val_acc": accuracy}

  def test_step(self, batch, batch_idx):
    events, targets = batch
    events = events.to(device)
    targets = targets.to(device)

    output = self.model(events)
    loss = self.loss(output["logits"], targets)
    accuracy = self.acc(output["probs"], targets)

    self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    self.log("test_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

    return {"test_loss": loss, "test_acc": accuracy}

  def predict_step(self, batch, batch_idx):
    events, _ = batch
    output = self.model(events)
    return torch.argmax(output["probs"], dim=1)

  def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=self.hparams.lr)
  
def train_setup(max_epochs, patience, log_dir, checkpoint_dir, name):
    trainer = L.Trainer(
        max_epochs=max_epochs,
        check_val_every_n_epoch=2,
        accelerator=device,
        devices="auto",
        precision="16-mixed",
        logger=L.loggers.TensorBoardLogger(save_dir=log_dir, version=name),
        log_every_n_steps=1,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
            ModelCheckpoint(dirpath=checkpoint_dir,
                            filename=name, save_top_k=1,
                            monitor="val_loss", mode="min")
        ]
    )

    return trainer
  
def train(model, train_loader, val_loader, trainer):
    trainer.fit(model, train_loader, val_loader)


def test(model, test_loader, trainer):
    trainer.test(model, test_loader)

  
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="A script that takes command-line arguments."
    )

    parser.add_argument(
        "--data-dir", type=str, required=True, help="The directory where the data is stored."
    )

    parser.add_argument(
        "--max-epochs", type=int, required=True, help="Set the max epochs."
    )

    parser.add_argument(
        "--name", type=str, required=True, help="Set the experiment name."
    )

    parser.add_argument(
        "--log-dir", type=str, required=True, help="The directory for the logs."
    )

    parser.add_argument(
        "--checkpoint-dir", type=str, required=True, help="The directory for the model checkpoints."
    )

    args = parser.parse_args()
    data_dir = args.data_dir
    max_epochs = args.max_epochs
    name = args.name
    log_dir = args.log_dir
    checkpoint_dir = args.checkpoint_dir

    dataset = NWLASLDataset(base_path=data_dir, transform=transform)

    train_data, val_data, test_data = random_split(dataset, [0.8, 0.1, 0.1])
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=PadTensors())
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, collate_fn=PadTensors())
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False, collate_fn=PadTensors())

    model = SignLanguageRecognition(lr=0.001, num_classes=100)

    trainer = train_setup(max_epochs=max_epochs, patience=3, log_dir=log_dir, checkpoint_dir=checkpoint_dir, name=name)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
