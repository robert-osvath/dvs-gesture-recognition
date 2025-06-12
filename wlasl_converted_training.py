import tonic
import tonic.transforms as TT
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm
from torchmetrics import Accuracy, ConfusionMatrix
import lightning.pytorch as L
import seaborn as sns
from torch.utils.tensorboard import SummaryWriter
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks.model_checkpoint import ModelCheckpoint
import json
import os
from utils.pad_tensors import PadTensors
import matplotlib.pyplot as plt
import argparse

#sensor_size = (128, 128, 2)
#transform = TT.ToFrame(sensor_size=sensor_size, time_window=10000)

device = "cuda" if torch.cuda.is_available() else "cpu"

class NWLASLDatasetTrain(Dataset):
  def __init__(self, train_path, transform=None):
    self.transform = transform
    self._base_path = train_path
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

    target = self.videos[idx]["target"]
    target = torch.tensor(target, dtype=torch.long)
    return events, target
  
class NWLASLDatasetVal(Dataset):
  def __init__(self, val_path, transform=None):
    self.transform = transform
    self._base_path = val_path
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
      # You could return None here and filter later, or handle the error
      # in a way that is appropriate for your use case.
      # For now, we'll re-raise after printing to identify the problematic file.
      raise RuntimeError(f"Error reading file: {video_path}")

    if self.transform:
      events = self.transform(events)

    target = self.videos[idx]["target"]
    target = torch.tensor(target, dtype=torch.long)
    return events, target
  
class NWLASLDatasetTest(Dataset):
  def __init__(self, test_path, transform=None):
    self.transform = transform
    self._base_path = test_path
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
      # You could return None here and filter later, or handle the error
      # in a way that is appropriate for your use case.
      # For now, we'll re-raise after printing to identify the problematic file.
      raise RuntimeError(f"Error reading file: {video_path}")

    if self.transform:
      events = self.transform(events)

    target = self.videos[idx]["target"]
    target = torch.tensor(target, dtype=torch.long)
    return events, target
  
class NormalizeEventTensor:
    def __init__(self, method='zscore'):
        self.method = method

    def __call__(self, x):
        x = torch.from_numpy(x.transpose(1, 0, 2, 3)).float()
        if self.method == 'zscore':
            mean = x.mean()
            std = x.std()
            return (x - mean) / (std + 1e-6)
        elif self.method == 'minmax':
            min_val = x.min()
            max_val = x.max()
            return (x - min_val) / (max_val - min_val + 1e-6)
        else:
            raise ValueError("Unsupported normalization method")
        
transform_1 = TT.Compose([
    TT.ToFrame(sensor_size=(128, 128, 2), time_window=10000),
    NormalizeEventTensor()
])

transform_2 = TT.Compose([
    TT.DropEvent(p=1/3),
    TT.ToFrame(sensor_size=(128, 128, 2), time_window=10000),
    NormalizeEventTensor()
])


class ASL3DConvNet(nn.Module):
  def __init__(self, classes):
    super(ASL3DConvNet, self).__init__()
    self.conv1 = self._make_conv_layer(2, 8)
    self.conv2 = self._make_conv_layer(8, 16)
    self.conv3 = self._make_conv_layer(16, 32)

    self.adaptive_pool = nn.AdaptiveAvgPool3d((6, 8, 8))

    self.fc1 = nn.Linear(32 * 6 * 8 * 8, 2048)
    self.fc2 = nn.Linear(2048, classes)

    self.dropout = nn.Dropout(0.5)

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
    x = self.dropout(F.relu(self.fc1(x)))
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
    self.cm = ConfusionMatrix(task="multiclass", num_classes=num_classes)

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
    self.cm(output["probs"], targets)

    self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
    self.log("test_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

    return {"test_loss": loss, "test_acc": accuracy}

  def on_test_epoch_end(self):
    confusion_matrix = self.cm.compute().cpu().numpy()

    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Labels")
    ax.set_ylabel("True Labels")
    ax.set_title("Confusion Matrix")

    writer = SummaryWriter(log_dir=self.logger.log_dir)
    writer.add_figure("Confusion Matrix", fig, global_step=self.current_epoch)
    writer.close()

    self.cm.reset()

  def predict_step(self, batch, batch_idx):
    events, _ = batch
    output = self.model(events)
    return torch.argmax(output["probs"], dim=1)

  def configure_optimizers(self):
    return optim.Adam(self.parameters(), lr=self.hparams.lr)
  
def setup_trainers(max_epochs, patience, name, log_dir, checkpoint_dir):
  trainer_1 =  L.Trainer(
    max_epochs=20,
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

  trainer_2 =  L.Trainer(
    max_epochs=20,
    check_val_every_n_epoch=2,
    accelerator=device,
    devices="auto",
    precision="16-mixed",
    logger=L.loggers.TensorBoardLogger(save_dir=log_dir, version=name),
    log_every_n_steps=1,
    callbacks=[
        EarlyStopping(monitor="val_loss", patience=patience, mode="min"),
        ModelCheckpoint(dirpath=checkpoint_dir,
                        filename=f'{name}2', save_top_k=1,
                        monitor="val_loss", mode="min")
    ]
  )

  return trainer_1, trainer_2

if __name__=="__main__":
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

  train_dir = os.path.join(data_dir, "train")
  val_dir = os.path.join(data_dir, "val")
  test_dir = os.path.join(data_dir, "test")

  train_dataset = NWLASLDatasetTrain(train_dir, transform=transform_1)
  val_dataset = NWLASLDatasetVal(val_dir, transform=transform_1)
  test_dataset = NWLASLDatasetTest(test_dir, transform=transform_1)

  train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=PadTensors())
  val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=PadTensors())
  test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=PadTensors())

  model = SignLanguageRecognition(lr=0.001, num_classes=20)

  trainer_1, trainer_2 = setup_trainers(max_epochs, 3, name, log_dir, checkpoint_dir)

  trainer_1.fit(model, train_loader, val_loader)
  trainer_1.test(model, test_loader)

  train_dataset = NWLASLDatasetTrain(train_dir, transform=transform_2)
  val_dataset = NWLASLDatasetVal(val_dir, transform=transform_2)
  test_dataset = NWLASLDatasetTest(test_dir, transform=transform_2)

  train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=PadTensors())
  val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=PadTensors())
  test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=PadTensors())

  model_2 = SignLanguageRecognition.load_from_checkpoint(os.path.join(checkpoint_dir, f'{name}.ckpt'))

  trainer_2.fit(model_2, train_loader, val_loader)
  trainer_2.test(model_2, test_loader)
