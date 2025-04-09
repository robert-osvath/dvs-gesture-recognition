import torch
import torch.nn as nn
import torch.nn.functional as F

class Gesture3DConvNet_v2(nn.Module):
    def __init__(self, classes):
        super(Gesture3DConvNet_v2, self).__init__()
        self.conv1 = self._make_conv_layer(2, 8)
        self.conv2 = self._make_conv_layer(8, 16)
        self.conv3 = self._make_conv_layer(16, 32)

        self.adaptive_pool = nn.AdaptiveAvgPool3d((12, 16, 16))

        self.fc1 = nn.Linear(32 * 12 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, classes)

    def _make_conv_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
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