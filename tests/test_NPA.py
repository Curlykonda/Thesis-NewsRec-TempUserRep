import unittest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tests.utils import generate_test_data

from source.models.NPA import NPA_wu
from source.my_datasets import DPG_Dataset

class MyTestCase(unittest.TestCase):
    def test_something(self):
        self.assertEqual(True, False)

class TestNPA_wu(unittest.TestCase):

    def test_model_build(self):
        # build small model
        model_params = {
            'emb_dim_user_id': 5,
            'emb_dim_pref_query': 5,
            'emb_dim_words': 5,
            'n_filters_cnn': 5,
            'dropout_p': 0.2
        }

        data_params = {
            'n_items': 100,
            'n_users': 10,
            'len_vocab': 10,
            'max_title_len': 5,
            'max_hist_len': 10
        }

        model = NPA_wu(data_params['n_users'], data_params['len_vocab'], pretrained_emb=None,
                       max_title_len=data_params['max_title_len'], **model_params)

        return model

    def test_forward_pass(self):
        data_params = {
            'n_items': 100,
            'n_users': 10,
            'len_vocab': 10,
            'max_title_len': 5,
            'max_hist_len': 10
        }


        # generate dummy data
        data_train, labels, vocab, news_as_word_ids = generate_test_data(**data_params)

        # build model
        model = self.test_model_build()

        #
        dataset = DPG_Dataset(data_train, labels, news_as_word_ids)
        data_gen = DataLoader(dataset, batch_size=2)

        for (hist_as_word_ids, cands_as_word_ids, u_id), labels in data_gen:
            preds = model.forward(hist_as_word_ids, cands_as_word_ids, u_id)


if __name__ == '__main__':
    suite = unittest.TestLoader().loadTestsFromTestCase(TestNPA_wu)
    unittest.TextTestRunner(verbosity=2).run(suite)
