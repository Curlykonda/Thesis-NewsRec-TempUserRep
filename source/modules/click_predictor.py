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
        # (B x user_rep_dim) * (B x news_rep_dim x n_candidates)
        # => (B x n_candidates)
        return torch.bmm(user_rep.unsqueeze(1), candidate_reps).squeeze()