import os
import numpy as np

from source.utils_npa import *
from source.metrics import *

import torch
import torch.nn as nn
import torch.functional as F


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

        # word embeddings
        self.word_embeddings = nn.Embedding(vocab_len, emb_dim_words, _weight=pretrained_emb)

        # news encoder
        self.news_encoder = CNN_wu(max_title_len*emb_dim_words, n_filters_cnn, emb_dim_pref_query)

        # preference query
        self.user_emb = UserEmbeddingLayer(n_users, emb_dim_user_id, emb_dim_pref_query)

        # user representation


        # click predictor

        pass

    def forward(self):

        pass

    def encode_news(self, news_articles_as_ids, user_id):

        emb_news = self.word_embeddings(news_articles_as_ids)

        pref_q_word, _ = self.user_emb(user_id)

        encoded_articles = self.news_encoder(emb_news, pref_q_word)

        return encoded_articles

class UserEmbeddingLayer(nn.Module):

    def __init__(self, n_users, emb_dim_user_id=50, dim_pref_query=200):
        super(UserEmbeddingLayer, self).__init__()

        self.n_users = n_users
        self.dim_pref_query = dim_pref_query

        self.user_embedding = nn.Embedding(n_users, emb_dim_user_id, padding_idx=0)
        self.dense_pref_word = nn.Linear(emb_dim_user_id, dim_pref_query) # NN #1
        self.dense_pref_article = nn.Linear(emb_dim_user_id, dim_pref_query) # NN #2

    def forward(self, user_id):
        u_emb = self.user_embedding(user_id)
        return F.relu(self.dense_pref_word(u_emb)), F.relu(self.dense_pref_article(u_emb))

class CNN_wu(nn.Module):

    def __init__(self, in_size, n_filters=400, dim_pref_q=200, kernel=3, stride=1, dropout_p=0.2):
        super(CNN_wu, self).__init__()

        self.cnn_encoder = nn.Sequential(
            nn.Conv1D(in_size, n_filters, kernel_size=kernel, stride=stride, padding_mode='same'),
                        F.relu,
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


class UserEncoder_wu(nn.Module):
    def __init__(self, dim_pref_q, dim_news_rep):
        super(UserEncoder_wu, self).__init__()

        self.dim_pref_q = dim_pref_q
        self.dim_news_rep = dim_news_rep

        self.proj_pref_q = nn.Sequential(
            nn.Linear(dim_pref_q, dim_news_rep),
            F.tanh()
        )

    def forward(self, contextual_news_rep, pref_q):



        pass

class PersonalisedAttention(nn.Module):
    def __init__(self, dim_pref_q, dim_news_rep):
        super(PersonalisedAttention, self).__init__()

        self.dim_pref_q = dim_pref_q
        self.dim_news_rep = dim_news_rep

        self.proj_pref_q = nn.Sequential(
            nn.Linear(dim_pref_q, dim_news_rep),
            F.tanh()
        )

        self.softmax = F.softmax()

    def forward(self, enc_input, pref_q):

        attn_a = torch.matmul(enc_input, self.proj_pref_q(pref_q))  # dimesions? Dot(2,1)?
        attn_weights = F.softmax(attn_a)
        attn_w_rep = (torch.matmul(attn_weights, enc_input)) # attention-weighted representation

        return attn_w_rep