import os
import numpy as np

from source.utils_npa import *
from source.metrics import *

from source.modules.click_predictor import SimpleDot


import torch
import torch.nn as nn
import torch.nn.functional as F


class NPA(torch.nn.Module):

    def __init__(self, news_encoder, user_encoder, click_predictor, dim_uid_emb, dim_pref_query, word_embeddings, dim_word_embeddings, cnn_filter, device):
        super(NPA, self).__init__()

        self.device = device

        self.news_encoder = news_encoder
        self.user_encoder = user_encoder
        self.click_predictor = click_predictor


        raise NotImplementedError()


    def forward(self, browsing_history, user_id, candidate_news):

        encoded_news = self.news_encoder.encode(browsing_history, user_id)

        user_representation = self.user_encoder.encode(encoded_news, user_id)

        encoded_candidates = self.news_encoder.encode(candidate_news, user_id)

        score = self.click_predictor(user_representation, encoded_candidates)

        return score

class NPA_wu(nn.Module):

    def __init__(self, n_users, vocab_len, pretrained_emb, emb_dim_user_id=50, emb_dim_pref_query=200,
                 emb_dim_words=300, max_title_len=30, n_filters_cnn=400, dropout_p=0.2, device='cpu'):
        super(NPA_wu, self).__init__()

        self.vocab_len = vocab_len
        self.max_title_len = max_title_len

        self.dim_news_rep = n_filters_cnn
        self.dim_user_rep = n_filters_cnn
        self.dim_emb_user_id = emb_dim_user_id
        self.dim_pref_q = emb_dim_pref_query

        if pretrained_emb:
            assert pretrained_emb.shape == [vocab_len, emb_dim_words]
            self.word_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_emb), freeze=False, padding_idx=0)      # word embeddings
        else:
            self.word_embeddings = nn.Embedding(vocab_len, emb_dim_words)

        self.user_id_embeddings = nn.Embedding(n_users, self.dim_emb_user_id, padding_idx=0)

        self.news_encoder = CNN_wu(in_size=(max_title_len*emb_dim_words), n_filters=n_filters_cnn, dim_pref_q=emb_dim_pref_query, dropout_p=dropout_p) # news encoder

        # preference queries
        self.pref_q_word = PrefQuery_wu(self.dim_pref_q, self.dim_emb_user_id)
        self.pref_q_article = PrefQuery_wu(self.dim_pref_q, self.dim_emb_user_id)

        self.user_encoder = PersonalisedAttention(emb_dim_pref_query, self.dim_news_rep) # user representation

        self.click_predictor = SimpleDot(self.dim_user_rep, self.dim_news_rep)  # click predictor

    def forward(self, brows_hist_as_ids, candidates_as_ids, user_id):

        brows_hist_reps = self.encode_news(brows_hist_as_ids, user_id)

        candidate_reps = self.encode_news(candidates_as_ids, user_id)

        user_rep = self.create_user_rep(brows_hist_reps, user_id)

        click_scores = self.click_predictor(user_rep, candidate_reps)

        return click_scores


    def encode_news(self, news_articles_as_ids, user_id):

        emb_news = self.word_embeddings(news_articles_as_ids)

        pref_q_word = self.pref_q_word(user_id)

        encoded_articles = self.news_encoder(emb_news, pref_q_word)

        return encoded_articles

    def create_user_rep(self, encoded_brows_hist, user_id):

        pref_q_article = self.pref_q_article(user_id)

        user_rep = self.user_encoder(encoded_brows_hist, pref_q_article)

        return user_rep

class PrefQuery_wu(nn.Module):
    '''
    Given an embedded user id, create a preference query vector (that is used in personalised attention)

    '''
    def __init__(self, dim_pref_query=200, dim_emb_u_id=50, activation='relu'):
        super(PrefQuery_wu, self).__init__()

        self.dim_pref_query = dim_pref_query
        self.dim_u_id = dim_emb_u_id

        self.lin_proj = nn.Linear(self.dim_u_id, self.dim_pref_query)

        assert activation in ['relu', 'tanh']

        if activation == 'relu':
            self.activation = nn.Tanh()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise KeyError()

    def forward(self, u_id):
        pref_query = self.lin_proj(u_id)

        return self.activation(pref_query)

class CNN_wu(nn.Module):

    def __init__(self, in_size, n_filters=400, dim_pref_q=200, kernel=3, stride=1, dropout_p=0.2):
        super(CNN_wu, self).__init__()

        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(in_size, n_filters, kernel_size=int(kernel), stride=stride, padding_mode='zeros'),
                        nn.ReLU(),
                        nn.Dropout(p=dropout_p)
        )

        self.pers_attn_word = PersonalisedAttention(dim_pref_q, n_filters)

    def forward(self, embedded_news, pref_query):
        contextual_rep = []
        # encode each browsed news article and concatenate
        for n_news in range(embedded_news.shape[1]):
            encoded_news = self.cnn_encoder(embedded_news[:, n_news, :])

            #pers attn
            contextual_rep.append(self.pers_attn_word(pref_query, encoded_news))

            #torch.bmm(A.view(6, 1, 256), B.view(6, 256, 1)) should do the trick! http://pytorch.org/docs/0.2.0/torch.html#torch.bmm

        return torch.cat(contextual_rep, axis=1)


class PersonalisedAttention(nn.Module):
    def __init__(self, dim_pref_q, dim_news_rep):
        super(PersonalisedAttention, self).__init__()

        self.dim_pref_q = dim_pref_q
        self.dim_news_rep = dim_news_rep

        self.proj_pref_q = nn.Sequential(
            nn.Linear(dim_pref_q, dim_news_rep),
            nn.Tanh()
        )

    def forward(self, enc_input, pref_q):

        attn_a = torch.matmul(enc_input, self.proj_pref_q(pref_q))  # dimesions? Dot(2,1)?
        attn_weights = F.softmax(attn_a)
        attn_w_rep = (torch.matmul(attn_weights, enc_input)) # attention-weighted representation

        return attn_w_rep