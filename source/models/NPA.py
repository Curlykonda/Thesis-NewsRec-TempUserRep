import os
import numpy as np

from utils_npa import *
from metrics import *

import torch
import torch.nn as nn
import torch.functional as F


class NPA(torch.nn.Module):

    def __init__(self, dim_uid_emb, dim_pref_query, word_embeddings, dim_word_embeddings, cnn_filter):
        super(NPA, self).__init__()

        raise NotImplementedError()


    def forward(self, *input, **kwargs):
        raise NotImplementedError()