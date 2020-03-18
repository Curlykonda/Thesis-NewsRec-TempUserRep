import numpy as np

import torch
import torch.nn as nn
import sys
sys.path.append("..")

from train_scripts.train_npa import try_var_loss_funcs

'''
Two things to test: 
1. compute losses with tensors on cuda to avoid sending them back to cpu
2. compare different loss functions

'''

def test_losses():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    batch_size = 100

    loss_fn = nn.BCEWithLogitsLoss()
    input = torch.randn(batch_size, requires_grad=True).to(device)
    target = torch.empty(batch_size).random_(2).to(device)
    loss = loss_fn(input, target)
    loss.backward()

if __name__ == "__main__":
    test_losses()