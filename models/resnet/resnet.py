import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)

        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

class ResNet3D(nn.Module):
    def __init__(self, num_blocks, num_classes, base_channels=16):
        super(ResNet3D, self).__init__()

        self.conv1 = nn.Conv3d(2, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(base_channels)
        self.relu = nn.ReLU(inplace=True)

        self.blocks = nn.ModuleList([BasicBlock3D(base_channels, base_channels)])
        in_channels = base_channels
        for _ in range(1, num_blocks):
            out_channels = in_channels * 2
            self.blocks.append(BasicBlock3D(in_channels, out_channels))
            in_channels = out_channels

        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.fc= nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        for block in self.blocks:
            x = block(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        x = self.fc(x)
        return {
            "logits": x,
            "probs" : F.softmax(x, dim=1)
        }
