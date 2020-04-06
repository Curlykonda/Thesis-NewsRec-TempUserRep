import torch
import torch.nn as nn
import torch.functional as F

from source.modules.attention import PersonalisedAttentionWu

class NewsEncoderWuCNN(nn.Module):

    def __init__(self, n_filters=400, dim_pref_q=200, word_emb_dim=300, kernel_size=3, dropout_p=0.2):
        super(NewsEncoderWuCNN, self).__init__()

        self.n_filters = n_filters
        self.dim_pref_q = dim_pref_q
        self.word_emb_dim = word_emb_dim

        self.dropout_in = nn.Dropout(p=dropout_p)

        self.cnn_encoder = nn.Sequential(nn.Conv1d(1, n_filters, kernel_size=(kernel_size, word_emb_dim), padding=(kernel_size - 2, 0)),
                        nn.ReLU(),
                        nn.Dropout(p=dropout_p)
        )

        self.pers_attn_word = PersonalisedAttentionWu(dim_pref_q, n_filters)

    def forward(self, embedded_news, pref_query):
        contextual_rep = []
        # embedded_news.shape = batch_size X max_hist_len X max_title_len X word_emb_dim
        embedded_news = self.dropout_in(embedded_news)
        # encode each browsed news article and concatenate
        for n_news in range(embedded_news.shape[1]):

            # concatenate words
            article_one = embedded_news[:, n_news, :, :].squeeze(1) # shape = (batch_size, title_len, emb_dim)

            # if n_news == 0:
            #     print(article_one.shape)

            encoded_news = self.cnn_encoder(article_one.unsqueeze(1))
            # encoded_news.shape = batch_size X n_cnn_filters X max_title_len

            #pers attn
            contextual_rep.append(self.pers_attn_word(encoded_news.squeeze(-1), pref_query))

            assert contextual_rep[-1].shape[1] == self.n_filters # batch_size X n_cnn_filters

        return torch.stack(contextual_rep, axis=2) # batch_s X dim_news_rep X history_len