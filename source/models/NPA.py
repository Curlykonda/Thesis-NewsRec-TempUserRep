import os
import numpy as np

import torch
import torch.nn as nn

from source.models.base import NewsRecBase
from source.modules.click_predictor import SimpleDot
from source.modules.news_encoder import NewsEncoderWuCNN
from source.modules.attention import PersonalisedAttentionWu

class VanillaNPANewsRec(NewsRecBase):

    def __init__(self, n_users, vocab_len, pretrained_emb, emb_dim_user_id=50, emb_dim_pref_query=200,
                 emb_dim_words=300, max_news_len=30, max_hist_len=50, d_news_rep=400, dropout_p=0.2, device='cpu'):
        super(VanillaNPANewsRec, self).__init__(n_users, vocab_len, pretrained_emb,
                                                emb_dim_user_id, emb_dim_pref_query, emb_dim_words,
                                                max_news_len, max_hist_len, d_news_rep, dropout_p,
                                                news_encoder=NewsEncoderWuCNN(n_filters=d_news_rep, word_emb_dim=emb_dim_words, dim_pref_q=emb_dim_pref_query, dropout_p=dropout_p),
                                                user_encoder=PersonalisedAttentionWu(emb_dim_pref_query, d_news_rep),
                                                interest_extractor=None,
                                                click_predictor=SimpleDot(dim_user_rep=d_news_rep, dim_cand_rep=d_news_rep),
                                                device=device)

    def create_user_rep(self, user_id, encoded_brows_hist):

        pref_q_article = self.pref_q_article(self.user_id_embeddings(user_id))
        self.user_rep = self.user_encoder(encoded_brows_hist, pref_q_article)

        return self.user_rep

    @classmethod
    def code(cls):
        return 'vanilla_npa'