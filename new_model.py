import torch
import torch.nn as nn
import torch.nn.functional as F

class Gesture3DConvNet_v2(nn.Module):
    def __init__(self, classes, nr_conv_layers):
        super(Gesture3DConvNet_v2, self).__init__()
        self.conv_layers = nn.ModuleList()

        for i in range(1, nr_conv_layers + 1):
            in_channels = 2 ** i
            out_channels = 2 ** (i + 1)
            self.conv_layers.append(self._make_conv_layer(in_channels, out_channels))

        final_output_channels = 2 ** (nr_conv_layers + 1)

        self.adaptive_pool = nn.AdaptiveAvgPool3d((12, 16, 16))

        self.fc1 = nn.Linear(final_output_channels * 12 * 16 * 16, 128)
        self.fc2 = nn.Linear(128, classes)

    def _make_conv_layer(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return {
            "logits": x,
            "probs": F.softmax(x, dim=1)
        }