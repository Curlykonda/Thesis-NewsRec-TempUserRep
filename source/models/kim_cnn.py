import numpy as np
import torch
import torch.nn as nn
import torch.functional as F
from torch.nn import Conv1d, MaxPool1d, Linear, ReLU, Dropout
from torch.nn import functional as F

# Utility function to calc output size
def output_size(in_size, kernel_size, stride, padding):
  output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
  return output//2

def mot_pooling(x):
  # Max-over-time pooling
  # X is shape n,c,w
  x = F.max_pool1d(x, kernel_size=x.shape[2])
  return x


class KimCNN(torch.nn.Module):
  # Shape after conv is (batch, x, y)
  def __init__(self, in_dim, conv_channels, num_classes):
    super(KimCNN, self).__init__()

    self.in_dim = in_dim
    self.conv_channels = []

    self.conv1 = Conv1d(in_channels=1, out_channels=100, kernel_size=3)
    self.conv2 = Conv1d(in_channels=1, out_channels=100, kernel_size=4)
    self.conv3 = Conv1d(in_channels=1, out_channels=100, kernel_size=5)

    self.fc = Linear(300, num_classes)

  def forward(self, x):
    # Pass through conv channels
    out1 = self.conv1(x)
    out2 = self.conv2(x)
    out3 = self.conv3(x)
    # Max over time pooling
    out1 = mot_pooling(out1)
    out2 = mot_pooling(out2)
    out3 = mot_pooling(out3)
    # Concatenate over channel dim
    out = torch.cat((out1, out2, out3), 1)
    # Flatten
    out = out.view(out.size(0), -1)
    # Pass to linear modules
    out = self.fc(out)

    return out
