import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score

from tests.utils import generate_test_data

from source.models.NPA import NPA_wu
from source.my_datasets import DPG_Dataset


#class TestNPA_wu(unittest.TestCase):


def test_model_build(data_params, model_params):
    # build small model
    model = NPA_wu(data_params['n_users'], data_params['len_vocab'], pretrained_emb=None,
                   max_title_len=data_params['max_title_len'], **model_params)

    return model

def test_forward_pass():

    model_params = {
        'emb_dim_user_id': 4,
        'emb_dim_pref_query': 5,
        'n_filters_cnn': 6,
        'emb_dim_words': 7,
        'dropout_p': 0.2
    }

    data_params = {
        'n_items': 100,
        'n_users': 11,
        'len_vocab': 12,
        'max_title_len': 8,
        'max_hist_len': 10
    }


    # generate dummy data
    data_train, labels, vocab, news_as_word_ids = generate_test_data(**data_params)

    # build model
    model = test_model_build(data_params, model_params)
    criterion = nn.BCEWithLogitsLoss()

    #
    batch_size = 3
    dataset = DPG_Dataset(data_train, labels, news_as_word_ids)
    data_gen = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for i_batch, sample in enumerate(data_gen): #(hist_as_word_ids, cands_as_word_ids, u_id), labels

        hist_as_word_ids, cands_as_word_ids, u_id = sample['input']
        lbls = sample['labels']

        if i_batch == 0:
            print(hist_as_word_ids.shape)
            print(lbls.shape)

        logits = model(hist_as_word_ids, cands_as_word_ids, u_id)

        loss1 = criterion(logits, lbls.float()) # or need to apply softmax to logits?
        loss1.backward()

        y_probs = torch.nn.functional.softmax(logits, dim=-1)
        #loss2 = criterion(y_probs, lbls.float())
        y_preds = y_probs.detach().argmax(dim=1)

        print(accuracy_score(lbls.argmax(dim=1), y_preds))

def test_conv1():
    kernel = 3
    emb_dim = 10
    seq_len = 5
    batch_size = 2
    n_filters = 20

    input = torch.randn(batch_size, seq_len, emb_dim)
    print(input.shape)
    conv1d = nn.Conv1d(1, n_filters, kernel_size=(kernel, emb_dim), padding=(kernel-2, 0))
    conv2d = nn.Conv2d(1, n_filters, kernel_size=(kernel, emb_dim), padding=(kernel-2, 0))

    out = conv1d(input.unsqueeze(1))
    print(out.shape)

    out2 = conv2d(input.unsqueeze(1))
    print(out2.shape)


if __name__ == '__main__':
    #suite = unittest.TestLoader().loadTestsFromTestCase(TestNPA_wu)
    #unittest.TextTestRunner(verbosity=2).run(suite)
    test_forward_pass()
    #test_conv1()
