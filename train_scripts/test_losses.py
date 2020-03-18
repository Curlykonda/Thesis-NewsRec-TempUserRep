import numpy as np

import torch
import torch.nn as nn
import sys
sys.path.append("..")

'''
Two things to test: 
1. compute losses with tensors on cuda to avoid sending them back to cpu
2. compare different loss functions

'''

def try_var_loss_funcs(logits, targets, i_batch):
    batch_size = logits.shape[0]

    softmax = nn.Softmax(dim=1)
    sigmoid = nn.Sigmoid()
    log_softm = nn.LogSoftmax(dim=1)

    bce = nn.BCELoss()
    bce_w_log = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    nll = nn.NLLLoss()
    acc = (logits.argmax(dim=1) == targets.argmax(dim=1)).sum().float().mean()

    softm_probs = softmax(logits)
    sigm_probs = sigmoid(logits)
    log_softm_probs = log_softm(logits)

    print("\n Batch {}".format(i_batch))
    print("TP acc {0:.3f}".format(acc))
    print("BCE softm {0:.3f} \t sigmoid {0:.3f}".format(bce(softm_probs, targets), bce(sigm_probs, targets)))
    print("BCE w Logits {0:.3f} \t softm {0:.3f}".format(bce_w_log(logits, targets), bce_w_log(softm_probs, targets)))
    print("CE softm {0:.3f} \t sigmoid {0:.3f}".format(ce(softm_probs, targets.argmax(dim=1)), ce(log_softm_probs, targets.argmax(dim=1))))
    print("NLL softm {0:.3f} \t log softm {0:.3f}".format(nll(softm_probs, targets.argmax(dim=1)), nll(log_softm_probs, targets.argmax(dim=1))))

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