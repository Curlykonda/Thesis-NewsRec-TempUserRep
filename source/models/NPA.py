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

        self.device = device
        self.vocab_len = vocab_len
        self.max_title_len = max_title_len

        self.dim_news_rep = n_filters_cnn
        self.dim_user_rep = n_filters_cnn
        self.dim_emb_user_id = emb_dim_user_id
        self.dim_pref_q = emb_dim_pref_query

        #representations
        self.user_rep = None
        self.brows_hist_reps = None
        self.candidate_reps = None
        self.click_scores = None

        if pretrained_emb is not None:
            #assert pretrained_emb.shape == [vocab_len, emb_dim_words]
            print("Emb shape is {} and should {}".format(pretrained_emb.shape, (vocab_len, emb_dim_words)))
            self.word_embeddings = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_emb), freeze=False, padding_idx=0)      # word embeddings
        else:
            self.word_embeddings = nn.Embedding(vocab_len, emb_dim_words)

        self.user_id_embeddings = nn.Embedding(n_users, self.dim_emb_user_id, padding_idx=0)

        self.news_encoder = CNN_wu(n_filters=n_filters_cnn, word_emb_dim=emb_dim_words, dim_pref_q=emb_dim_pref_query, dropout_p=dropout_p) # news encoder

        # preference queries
        self.pref_q_word = PrefQuery_wu(self.dim_pref_q, self.dim_emb_user_id)
        self.pref_q_article = PrefQuery_wu(self.dim_pref_q, self.dim_emb_user_id)

        self.user_encoder = PersonalisedAttention(emb_dim_pref_query, self.dim_news_rep) # user representation

        self.click_predictor = SimpleDot(self.dim_user_rep, self.dim_news_rep)  # click predictor

    def forward(self, user_id, brows_hist_as_ids, candidates_as_ids):

        brows_hist_reps = self.encode_news(user_id, brows_hist_as_ids)
        self.brows_hist_reps = brows_hist_reps

        candidate_reps = self.encode_news(user_id, candidates_as_ids)
        self.candidate_reps = candidate_reps

        user_rep = self.create_user_rep(user_id, brows_hist_reps)

        click_scores = self.click_predictor(user_rep, candidate_reps)
        self.click_scores = click_scores

        #self.get_representation_shapes()

        return click_scores


    def encode_news(self, user_id, news_articles_as_ids):

        # (B x hist_len x art_len) -> (vocab_len x emb_dim_word)
        # => (B x hist_len x art_len x emb_dim_word)
        emb_news = self.word_embeddings(news_articles_as_ids) # assert dtype == 'long'

        pref_q_word = self.pref_q_word(self.user_id_embeddings(user_id))

        encoded_articles = self.news_encoder(emb_news, pref_q_word)

        return encoded_articles

    def create_user_rep(self, user_id, encoded_brows_hist):

        pref_q_article = self.pref_q_article(self.user_id_embeddings(user_id))

        user_rep = self.user_encoder(encoded_brows_hist, pref_q_article)

        self.user_rep = user_rep

        return user_rep

    def get_representation_shapes(self):
        shapes = {}
        shapes['user_rep'] = self.user_rep.shape if self.user_rep != None else None
        shapes['brow_hist'] = self.brows_hist_reps.shape
        shapes['cands'] = self.candidate_reps.shape
        shapes['scores'] = self.click_scores.shape

        for key, shape in shapes.items():
            print("{} shape {}".format(key, shape))

        return

class PrefQuery_wu(nn.Module):
    '''
    Given an embedded user id, create a preference query vector (that is used in personalised attention)

    '''
    def __init__(self, dim_pref_query=200, dim_emb_u_id=50, activation='relu', device='cpu'):
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
        #print(u_id.shape)
        pref_query = self.lin_proj(u_id) # batch_size X u_id_emb_dim

        return self.activation(pref_query)

class CNN_wu(nn.Module):

    def __init__(self, n_filters=400, dim_pref_q=200, word_emb_dim=300, kernel_size=3, dropout_p=0.2):
        super(CNN_wu, self).__init__()

        self.n_filters = n_filters
        self.dim_pref_q = dim_pref_q
        self.word_emb_dim = word_emb_dim

        self.cnn_encoder = nn.Sequential(nn.Conv1d(1, n_filters, kernel_size=(kernel_size, word_emb_dim), padding=(kernel_size - 2, 0)),
                        nn.ReLU(),
                        nn.Dropout(p=dropout_p)
        )

        self.pers_attn_word = PersonalisedAttention(dim_pref_q, n_filters)

    def forward(self, embedded_news, pref_query):
        contextual_rep = []
    # embedded_news.shape = batch_size X max_hist_len X max_title_len X word_emb_dim

        # encode each browsed news article and concatenate
        for n_news in range(embedded_news.shape[1]):

            # concatenate words
            article_one = embedded_news[:, n_news, :, :].squeeze() # shape = (batch_size, title_len, emb_dim)

            encoded_news = self.cnn_encoder(article_one.unsqueeze(1))
            # encoded_news.shape = batch_size X n_cnn_filters X max_title_len

            #pers attn
            contextual_rep.append(self.pers_attn_word(encoded_news.squeeze(), pref_query))

            assert contextual_rep[-1].shape[1] == self.n_filters # batch_size X n_cnn_filters

        return torch.stack(contextual_rep, axis=2) # batch_s X dim_news_rep X history_len


class PersonalisedAttention(nn.Module):
    def __init__(self, dim_pref_q, dim_news_rep):
        super(PersonalisedAttention, self).__init__()

        self.dim_pref_q = dim_pref_q
        self.dim_news_rep = dim_news_rep

        self.proj_pref_q = nn.Sequential(
            nn.Linear(dim_pref_q, dim_news_rep),
            nn.Tanh()
        )

        self.attn_weights = None

    def forward(self, enc_input, pref_q):

        # enc_input.shape = (batch_size, dim_news_rep, title_len)

        pref_q = self.proj_pref_q(pref_q) # transform pref query

        attn_a = torch.bmm(torch.transpose(enc_input, 1, 2), pref_q.unsqueeze(2)).squeeze() # dot product over batch http://pytorch.org/docs/0.2.0/torch.html#torch.bmm
        attn_weights = F.softmax(attn_a, dim=-1)

        self.attn_weights = attn_weights
        #assert torch.sum(attn_weights, dim=1) == torch.ones(attn_weights.shape[0], dtype=float) # (normalised) attn weights should sum to 1

        attn_w_rep = torch.matmul(enc_input, attn_weights.unsqueeze(2)).squeeze() # attn-weighted representation r of i-th news

        return attn_w_rep