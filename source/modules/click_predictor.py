import numpy as np

import torch
import torch.nn as nn
import torch.functional as F

class SimpleDot(nn.Module):
    def __init__(self, dim_user_rep, dim_cand_rep):
        super(SimpleDot, self).__init__()

        self.dim_user_rep = dim_user_rep
        self.dim_cand_rep = dim_cand_rep

        assert dim_user_rep == dim_cand_rep


    def forward(self, user_rep, candidate_reps):
        return torch.matmul(user_rep, candidate_reps)