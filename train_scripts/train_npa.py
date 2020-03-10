

import numpy as np
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split

from source.my_datasets import DPG_Dataset
from source.models.NPA import NPA_wu
from utils_npa import get_dpg_data, get_embeddings_from_pretrained

import source.utils_npa
import source.utils

def train(config):

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_params = { 'batch_size': config.batch_size,
                     'shuffle': True}

    #set random seeds
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    # data is indexed by user_ids
    '''
    data is first processed by functions in 'utils_npa' which follow the structure from original Wu NPA code
    
    '''
    dataset, labels, vocab, news_as_word_ids, art_id2idx, u_id2idx = get_dpg_data(config.article_data, config.user_data, config.neg_sample_ratio, config.max_hist_len, max_len_news_title=30)

    word_embeddings = get_embeddings_from_pretrained(vocab, emb_path=config.word_emb_path)

    train_data, test_data = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=config.random_seed)

    train_dataset = DPG_Dataset(train_data, labels, news_as_word_ids)
    train_generator = DataLoader(train_dataset, config.batch_size)

    # build model
    npa_model = NPA_wu(n_users=len(dataset), vocab_len=len(vocab), pretrained_emb=word_embeddings,
                       emb_dim_user_id=50, emb_dim_pref_query=200, emb_dim_words=300, max_title_len=config.max_hist_len)
    npa_model.to(device)

    criterion = nn.BinaryCrossEntropy()
    #optim

    for epoch in range(config.n_epochs):
        npa_model.train()
        for (history, candidates, user_ids), labels in train_generator:

            labels = labels.to(device)

            click_scores = npa_model(history.to(device), candidates.to(device), user_ids.to(device))

            # compute loss

            # optimiser backward


if __name__ == "__train__":
    parser = argparse.ArgumentParser()

    #general
    parser.add_argument('--random_seed', type=int, default=42,
                        help='random seed for reproducibility')

    # input data
    parser.add_argument('--data_type', type=str, default='DPG',
                        help='options for data format: DPG, NPA or Adressa ')
    parser.add_argument('--data_path', type=str, default='../datasets/dpg/i10k_u5k_s30/',
                        help='path to data directory')

    parser.add_argument('--article_ids', type=str, default='../datasets/dpg/i10k_u5k_s30/item_ids.pkl',
                        help='path to directory with item id pickle file')
    parser.add_argument('--article_data', type=str, default='../datasets/dpg/i10k_u5k_s30/news_data.pkl',
                        help='path to article data pickle file')
    parser.add_argument('--user_data', type=str, default='../datasets/dpg/i10k_u5k_s30/user_data.pkl',
                        help='path to user data pickle file')

    parser.add_argument('--word_emb_path', type=str, default='../embeddings/glove_eng.840B.300d.txt',
                        help='path to directory with word embeddings')

    parser.add_argument('--pkl_path', type=str, default='../datasets/books-pickle/', help='path to save pickle files')

    # preprocessing
    parser.add_argument('--max_hist_len', type=int, default=50,
                        help='maximum length of user reading history, shorter ones are padded; should be in accordance with the input datasets')

    parser.add_argument('--neg_sample_ratio', type=int, default=4,
                        help='Negative sample ratio N: for each positive impression generate N negative samples')

    #training
    parser.add_argument('--batch_size', type=int, default=100,
                        help='batch size for training')

    config = parser.parse_args()

    train(config)