import torch
import torch.nn as nn
import torch.nn.functional as F
import snntorch as snn
from snntorch import utils, surrogate 

class Gesture3DCSNN(nn.Module):
  def __init__(self, beta, classes):
    super(Gesture3DCSNN, self).__init__()
    self.conv1 = self._make_conv_layer(2, 8, beta=beta)
    self.conv2 = self._make_conv_layer(8, 16, beta=beta)
    self.conv3 = self._make_conv_layer(16, 32, beta=beta)

    self.adaptive_pool = nn.AdaptiveAvgPool3d((12, 16, 16))

    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(32 * 16 * 16, 128),
        snn.Leaky(beta=0.9, spike_grad=surrogate.atan(), init_hidden=True),
        nn.Linear(128, classes),
        snn.Leaky(beta=0.9, spike_grad=surrogate.atan(), init_hidden=True, output=True)
    )

  def _make_conv_layer(self, in_channels, out_channels, beta, spike_grad=surrogate.atan()):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        snn.Leaky(beta=beta, spike_grad=spike_grad, init_hidden=True),
        nn.MaxPool3d(kernel_size=2, stride=2)
    )

  def forward(self, x):
    utils.reset(self.conv1)
    utils.reset(self.conv2)
    utils.reset(self.conv3)

    # 3d convolutions for feature extraction
    # [batch, time, channels, height, width] -> [batch, channels, time, height, width]
    x = x.permute(0, 2, 1, 3, 4)
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.adaptive_pool(x)
    
    # rearrange the output tensor to have the shape of [time_steps, batch_size, channels, height, width]
    # [batch, channels, time, height, width] -> [time, batch, channels, height, width]
    x = x.permute(2, 0, 1, 3, 4)

    # we need to get a final shape of [time_steps, batch_size, num_classes]
    # the loss function relies on spike counts over time steps, so we could iterate throught the time steps
    spike_rec = []
    mem_rec = []
    time_steps = x.shape[0]
    for t in range(time_steps):
      spike_out, mem_out = self.classifier(x[t])
      spike_rec.append(spike_out)
      mem_rec.append(mem_out)
      utils.reset(self.classifier)

    out_spikes = torch.stack(spike_rec)
    out_mems = torch.stack(mem_rec)
    probs = F.softmax(torch.sum(out_spikes, dim=0), dim=1)

    return {'spike_rec': out_spikes, 'mem_rec': out_mems, 'probs': probs}