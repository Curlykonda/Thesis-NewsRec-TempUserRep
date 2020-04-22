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
  def __init__(self, n_filters, word_emb_dim, kernels=[3, 4, 5]):
    super(KimCNN, self).__init__()

    self.n_filters = n_filters
    self.word_emb_dim = word_emb_dim
    self.kernels = kernels
    self.convs = nn.ModuleList([
      nn.Conv1d(1, out_channels=n_filters, kernel_size=(kernel, word_emb_dim), padding=(kernel-2, 0))
      for kernel in kernels
    ])

  def forward(self, x):
    # Pass through each conv layer
    outs = [conv(x) for conv in self.convs]

    # Max over time pooling
    outs_pooled = [mot_pooling(out) for out in outs]
    # Concatenate over channel dim
    out = torch.cat(outs_pooled, 1)
    # Flatten
    out = out.view(out.size(0), -1)

    return out
