import os
import numpy as np
import torch
import random

from source.utils_npa import get_labels_from_data

def generate_test_data(n_items=100, n_users=10, len_vocab=10, max_title_len=5, max_hist_len=6, **kwargs):

    #dataset[u_id] = [user_id, hist, candidates]
    item_ids = list(range(n_items))
    u_ids = list(range(n_users))
    vocab = list(range(len_vocab)) # usually vocab is a dictionary mapping word to index. but here it's just a list

    dataset = []

    news_as_word_ids = [random.sample(vocab, max_title_len) for _ in item_ids]
    news_as_word_ids = np.array(news_as_word_ids)
    lbls = [0, 1]

    for u_id in u_ids:

        hist = random.sample(item_ids, max_hist_len)

        cands = random.sample(item_ids, 2)
        random.shuffle(lbls)

        dataset.append({'u_id': u_id, 'history': hist, 'candidates': cands, 'labels': lbls})

    return dataset, get_labels_from_data(dataset), vocab, news_as_word_ids

def check_model(model, model_name, x, y, check_model_io=True):
    #compile model,train and evaluate it,then save/load weight and model file.

    pass