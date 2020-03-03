import os
import pickle
import numpy as np
import random

from collections import defaultdict
from nltk.tokenize import sent_tokenize

from sklearn.metrics.pairwise import cosine_similarity

import torch
import torch.nn as nn

from transformers import BertTokenizer, BertModel


def get_dummy_bert_output(self, batch_size, dim_bert=64, seq_len=20, n_layers=12, hidden_outs=True, attn_weights=True):
    bert_outs = []
    # last_hidden, pooled_out, hidden_outs, attention_weights = bert_output
    # 1. last_hidden
    last_hidden = torch.randn([batch_size, seq_len, dim_bert])
    bert_outs.append(last_hidden)  # batch_size x seq_len x dim_bert

    # 2. pooled_out
    bert_outs.append(torch.randn([batch_size, dim_bert]))

    # 3. hidden_outs
    if hidden_outs:
        h_outs = [torch.randn([batch_size, seq_len, dim_bert]) for n in range(n_layers)]
        h_outs.append(last_hidden)
        bert_outs.append(h_outs)

    # 4. attn_weights
    if attn_weights:
        a_weights = [torch.randn([batch_size, seq_len, dim_bert]) for n in range(n_layers)]
        bert_outs.append(a_weights)

    return bert_outs


class BertFeatureExtractor():

    def __init__(self, device, batch_size, max_seq_len, **kwargs):
        super().__init__(**kwargs)

        self.device = device
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        self.dev_mode = False
        self.batch_size_cur = batch_size
        self.max_seq_len_cur = max_seq_len
        self.embedding_dim_cur = 768

        self.sent_tokenizer = sent_tokenize
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_tokenizer.add_special_tokens({"unk_token": '[UNK]', 'cls_token': '[CLS]',
                                                'pad_token': '[PAD]', 'sep_token': '[SEP]'})

        self.bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True,
                                                    output_attentions=True)
        self.bert_model.resize_token_embeddings(len(self.bert_tokenizer))
        self.bert_model.to(device)

        self._feat_methods = ['pooled_out', 'last_cls', 'pool_all_last', 'pool_cls_n', 'pool_last_n', 'sum_last_n',
                              'sum_all']
        self._naive_feat_methods = ['naive_mean', 'naive_sum']

        # self.word_embeddings =

    @property
    def _get_feat_methods(self):
        return self._feat_methods

    @property
    def _get_naive_feat_methods(self):
        return self._naive_feat_methods

    def __call__(self, *args, **kwargs):
        return self.bert_model(*args, **kwargs)

    def get_naive_seq_emb(self, tokens, method='naive_mean'):

        if method not in self._get_naive_feat_methods():
            raise KeyError("'{}' is not a valid method!".format(method))

        word_embeddings = self.bert_model.get_input_embeddings()
        word_embeddings.to(self.device)
        tokens_emb = word_embeddings(torch.tensor(tokens, device=self.device)).to(self.device)
        if method == 'naive_mean':
            emb_out = torch.mean(tokens_emb, dim=1)
        elif method == 'naive_sum':
            emb_out = torch.sum(tokens_emb, dim=1)
        else:
            raise NotImplementedError()

        return emb_out

    def get_bert_word_embeddings(self):
        return self.bert_model.get_input_embeddings().to(self.device)

    def extract_bert_features(self, bert_output, method, n_layers):
        """

        :param bert_output:
        :param method:
        :param n_layers:
        :return: torch tensor containing the extracted features of shape batch_size X emb_dim (768)
        """

        assert len(bert_output) == 4

        if method not in self._get_feat_methods():
            raise KeyError("'{}' is not a valid method!".format(method))

        last_hidden, pooled_out, hidden_outs, attention_weights = bert_output

        if 'pooled_out' == method:
            x_out = pooled_out  # batch_size x dim_emb
        elif 'last_cls' == method:
            # take the embedding of CLS token of last hidden layer
            x_out = last_hidden[:, 0, :]  # batch_size x dim_emb
        elif 'pool_all_last' == method:
            # average embeddings of last hidden layer of all tokens
            x_out = torch.mean(last_hidden, dim=1)
        elif 'pool_cls_n' == method and n_layers:
            x_out = torch.mean(torch.cat([hidden[:, 0, :].unsqueeze(1) for hidden in hidden_outs[-n_layers:]], dim=1),
                               dim=1)
        elif 'pool_last_n' == method and n_layers:
            # average embeddings of last N hidden layers of all tokens
            x_out = torch.mean(torch.cat(hidden_outs[-n_layers:], dim=1), dim=1)
        elif 'sum_last_n' == method and n_layers:
            # sum embeddings of last N hidden layers of all tokens
            x_out = torch.sum(torch.cat(hidden_outs[-n_layers:], dim=1), dim=1)
            # sum last four hidden => 95.9 F1 on dev set for NER
        elif 'sum_all' == method:
            x_out = torch.sum(torch.cat(hidden_outs, dim=1), dim=1)
        else:
            raise NotImplementedError()

        # print(method)
        # print(x_out.shape)
        return x_out

    def test_feature_extraction(self, n_items, methods, batch_size, emb_dim, seq_len, **kwargs):

        self.dev_mode = True
        #save default properties



        dummy_content = [0] * n_items
        self.embedding_dim_cur = emb_dim
        self.batch_size_cur = batch_size

        encoded_text, n_words = self.encode_text_to_features(dummy_content, methods, max_seq_len=seq_len)

        #restore default properties
        self.dev_mode = False

        return encoded_text

    def encode_text_to_features(self, content, methods, count_all_words=True, max_seq_len=None, add_special_tokens=True,
                                lower_case=True, naive_method='naive_mean'):

        encoded_text = defaultdict(dict)
        encodings = defaultdict(dict)

        if count_all_words:
            n_words_per_text = defaultdict(dict)

        if max_seq_len:
            self.max_seq_len_cur = max_seq_len
        else:
            self.max_seq_len_cur = self.max_seq_len

        start_idx = 0
        stop_idx = self.batch_size
        n_iterations = 0  # loops: n_methods * n_keys
        n_items = len(content)

        all_item_keys = range(n_items)
        slice_idx = list(range(0, self.batch_size))

        while (start_idx < n_items):

            # handling edge cases
            if stop_idx > n_items:
                # indices and slice range
                slice_idx = list(range(0, (n_items - start_idx)))
                stop_idx = n_items

            # divide item_inds into batches
            item_keys = range(start_idx, stop_idx)

            # assert len(item_keys) == batch_size
            if len(item_keys) != self.batch_size:
                print(start_idx)
                print(stop_idx)
                print(len(slice_idx))

            if not self.dev_mode:
                #tokenise batch of text
                tokens, n_words = zip(
                    *[self.tokenize_text_to_ids(text, add_special_tokens, lower_case) for text
                      in content[item_keys]]) #list(df['reviewText'])

                if naive_method:
                    tokens_raw, _ = zip(*[self.tokenize_text_to_ids(text, add_special_tokens=False, lower_case=True)
                                          for text in content[item_keys]])

                #create tensor from tokens
                x_in = torch.tensor(tokens, requires_grad=False, device=self.device).long()  # batch_size x max_len

                # create naive sequence features
                if naive_method:
                    naive_emb = self.get_naive_seq_emb(tokens_raw, method=naive_method) #batch_size x emb_dim

                # generate BERT output
                bert_outputs = self.encode_input_ids(x_in)

            else:
                # create naive sequence features
                naive_emb = torch.randn([self.batch_size, self.embedding_dim_cur])

                # generate dummy BERT output
                bert_outputs = get_dummy_bert_output(len(slice_idx), self.embedding_dim_cur, self.max_seq_len_cur)

            # extract BERT features & add features to diotionary "encoded_text"
            for (method, n) in methods.items():
                encodings[method] = self.extract_bert_features(bert_outputs, method, n)  # batch_size x emb_dim

                assert encodings[method].shape[0] == len(slice_idx)
                # print(encodings[method].shape[0])
                # print(len(slice_idx))

                # add features to dictionary
                for idx in slice_idx:
                    if naive_method not in encoded_text[item_keys[idx]]:
                        encoded_text[item_keys[idx]][naive_method] = naive_emb[idx, :]  # add naive features

                    encoded_text[item_keys[idx]][method] = encodings[method][idx, :]
                    n_iterations += 1

            # assign word count to corresponding item key
            if count_all_words:
                n_words_per_text = {**n_words_per_text, **dict(zip(item_keys, n_words))}

            start_idx += self.batch_size
            stop_idx += self.batch_size

        print("Iterations: {}".format(n_iterations))

        return encoded_text, n_words

    def truncate_seq(self, tokens):

        max_len = self.max_seq_len_cur

        if len(tokens) < max_len:
            tokens = tokens[:-1]
            n = max_len - len(tokens) - 1
            tokens += n * [self.bert_tokenizer.pad_token]
        elif len(tokens) > max_len:
            tokens = tokens[:max_len - 1]
        else:
            return tokens

        tokens.append(self.bert_tokenizer.sep_token)

        return tokens

    def tokenize_text_to_ids(self, text, add_special_tokens=True, lower_case=False):
        """
        With tokenizer, separate text first into tokens
        and then convert these to corresponding IDs of the vocabulary

        Return:
            tokens: list of token IDs
            n_words: number of words in full sequence (before truncating)
        """
        sents = self.sent_tokenizer(text)
        tokens = []
        n_words = 0
        added_tokens = 0

        max_len = self.max_seq_len_cur

        if add_special_tokens:
            tokens.append(self.bert_tokenizer.cls_token)
            added_tokens += 1

        # split each sentence of the text into tokens
        for s in sents:
            if lower_case:
                tokens.extend([word.lower() for word in self.bert_tokenizer.tokenize(s) if word.isalpha()])
            else:
                tokens.extend([word for word in self.bert_tokenizer.tokenize(s) if word.isalpha()])

            if add_special_tokens:
                tokens.append(self.bert_tokenizer.sep_token)
                added_tokens += 1

        n_words = len(tokens) - added_tokens

        tokens = self.truncate_seq(tokens)

        #assert len(tokens) == max_len

        return self.bert_tokenizer.convert_tokens_to_ids(tokens), n_words

    def encode_input_ids(self, x_in):
        """
        Forward pass through BertModel with input tokens to produce Bert Outputs (last_hidden, pooled_out, etc.)

        :param x_in: torch tensor of shape batch_size x max_seq_len
        :return: list of torch tensor (bert_output): last_hidden, pooled_out, hidden_outs, attention_weights
        """
        return self.bert_model(x_in)