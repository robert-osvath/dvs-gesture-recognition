import torch
import torch.nn as nn
import torch.nn.functional as F

class Gesture3DConvNet(nn.Module):
  def __init__(self, input_shape, classes):
    super(Gesture3DConvNet, self).__init__()
    self.conv1 = self._make_conv_layer(2, 8)
    self.conv2 = self._make_conv_layer(8, 16)
    self.conv3 = self._make_conv_layer(16, 32)

    self.input_shape = input_shape
    self.flattened_size = self._infer_flattened_size(input_shape)

    self.fc1 = nn.Linear(self.flattened_size, 128)
    self.fc2 = nn.Linear(128, classes)

  def _make_conv_layer(self, in_channels, out_channels):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.MaxPool3d(kernel_size=2, stride=2)
    )

  def _infer_flattened_size(self, input_shape):
    dummy_input = torch.zeros(1, *input_shape)
    x = self.conv1(dummy_input)
    x = self.conv2(x)
    x = self.conv3(x)
    flattened = x.view(1, -1)
    return flattened.size(1)

  def forward(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = x.view(-1, self.flattened_size)
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return {
        "logits": x,
        "probs": F.softmax(x, dim=1)
    }
  