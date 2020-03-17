import numpy as np
import argparse
import pickle
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import sys
sys.path.append("..")

from source.my_datasets import DPG_Dataset
from source.models.NPA import NPA_wu
from source.utils_npa import get_dpg_data, get_embeddings_from_pretrained

import source.utils_npa
import source.utils

def train(config):

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_params = { 'batch_size': config.batch_size,
                     'shuffle': True}

    hyper_params = {'lr': None, 'neg_sample_ratio': None} # TODO: aggregate h_params for SummaryWriter

    #set random seeds
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    # data is indexed by user_ids
    '''
    data is first processed by functions in 'utils_npa' which follow the structure from original Wu NPA code
    
    '''
    dataset, vocab, news_as_word_ids, art_id2idx, u_id2idx = get_dpg_data(config.data_path, config.neg_sample_ratio,
                                                                          config.max_hist_len, config.max_news_len)

    word_embeddings = get_embeddings_from_pretrained(vocab, emb_path=config.word_emb_path)

    train_data, test_data = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=config.random_seed)

    train_dataset = DPG_Dataset(train_data, news_as_word_ids)
    train_generator = DataLoader(train_dataset, config.batch_size)
    print("Train on {} samples".format(train_dataset.__len__()))

    test_dataset = DPG_Dataset(test_data, news_as_word_ids)
    test_generator = DataLoader(test_dataset, config.batch_size)

    # build model
    npa_model = NPA_wu(n_users=len(dataset), vocab_len=len(vocab), pretrained_emb=word_embeddings,
                       emb_dim_user_id=50, emb_dim_pref_query=200, emb_dim_words=300, max_title_len=config.max_hist_len)
    npa_model.to(device)

    #optim
    if config.bce_logits:
        #crit_bce_logits = nn.BCEWithLogitsLoss()
        criterion = nn.BCEWithLogitsLoss()
    else:
        #crit_bce = nn.BCELoss()
        criterion = nn.BCELoss()
    optim = torch.optim.Adam(npa_model.parameters(), lr=0.001)


    writer = SummaryWriter(config.results_path) # logging

    acc = {'train': [], 'test': []}
    losses = {'train': [], 'test': []}
    print_shapes = True
    DEBUG = False

    for epoch in range(config.n_epochs):
        t0 = time.time()
        npa_model.train()

        acc_ep = []
        loss_ep = []

        for i_batch, sample in enumerate(train_generator):  # (hist_as_word_ids, cands_as_word_ids, u_id), labels
            npa_model.zero_grad()
            user_ids, brows_hist, candidates = sample['input']
            lbls = sample['labels']
            lbls = lbls.float()

            # forward pass
            logits = npa_model(user_ids.long().to(device), brows_hist.long().to(device), candidates.long().to(device))
            if print_shapes:
                npa_model.get_representation_shapes()
                print_shapes = False

            y_probs = torch.nn.functional.softmax(logits, dim=-1)
            y_preds = y_probs.detach().cpu().argmax(axis=1)


            # compute loss
            if config.bce_logits:
                loss_bce = criterion(logits.cpu(), lbls.cpu()) #loss_bce_logits = crit_bce_logits(logits.cpu(), lbls.cpu())  # or need to apply softmax to logits?
            else:
                loss_bce = criterion(y_probs.cpu(), lbls.cpu()) #loss_bce = crit_bce(y_probs.cpu(), lbls.cpu())

            # optimiser backward
            optim.zero_grad()
            loss_bce.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
            optim.step()

            acc_ep.append(accuracy_score(lbls.argmax(dim=1), y_preds))
            loss_ep.append(loss_bce.item())


            if DEBUG:
                break

        # logging
        t1 = time.time()
        losses['train'].append(np.mean(loss_ep))
        acc['train'].append(np.mean(acc_ep))
        writer.add_scalar('Loss/Train', np.mean(loss_ep), epoch)
        writer.add_scalar('Acc/Train', np.mean(acc_ep), epoch)
        writer.add_scalar('Acc-Var/Train', np.var(acc_ep), epoch)
        writer.add_scalar('Loss-Var/Train', np.var(loss_ep), epoch)

        #evaluate on test set
        acc_ep = []
        loss_ep = []

        npa_model.eval()

        for sample in test_generator:
            user_ids, brows_hist, candidates = sample['input']
            lbls = sample['labels']
            lbls = lbls.float()

            # forward pass
            with torch.no_grad():
                logits = npa_model(user_ids.long().to(device), brows_hist.long().to(device),
                                   candidates.long().to(device))

            y_probs = torch.nn.functional.softmax(logits, dim=-1)
            y_preds = y_probs.detach().cpu().argmax(dim=1)

            # compute loss
            if config.bce_logits:
                test_loss = criterion(logits.cpu(), lbls.cpu())
            else:
                test_loss = criterion(y_probs.cpu(), lbls.cpu())

            acc_ep.append(accuracy_score(lbls.argmax(dim=1), y_preds))
            loss_ep.append(test_loss.item())

            if DEBUG:
                break

        # logging
        t2 = time.time()
        losses['test'].append(np.mean(loss_ep))
        acc['test'].append(np.mean(acc_ep))
        writer.add_scalar('Loss/Test', np.mean(loss_ep), epoch)
        writer.add_scalar('Acc/Test', np.mean(acc_ep), epoch)
        writer.add_scalar('Acc-Var/Test', np.var(acc_ep), epoch)
        writer.add_scalar('Loss-Var/Test', np.var(loss_ep), epoch)

        print("{} epoch:".format(epoch))
        print("TRAIN: acc {:0.3f} \t BCE loss {:1.3f} in {:0.1f}s".format(np.mean(acc_ep), np.mean(loss_ep), (t1-t0)))
        print("TEST: acc {:0.3f} \t BCE loss {:1.3f} in {:0.1f}s".format(np.mean(acc_ep), np.mean(loss_ep), (t2-t1)))

    #writer.add_figure()
    #write.add_hparams

if __name__ == "__main__":
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

    parser.add_argument('--results_path', type=str, default='../results/', help='path to save metrics')

    # preprocessing
    parser.add_argument('--max_hist_len', type=int, default=50,
                        help='maximum length of user reading history, shorter ones are padded; should be in accordance with the input datasets')
    parser.add_argument('--max_news_len', type=int, default=30,
                        help='maximum length of news article, shorter ones are padded; should be in accordance with the input datasets')

    parser.add_argument('--neg_sample_ratio', type=int, default=4,
                        help='Negative sample ratio N: for each positive impression generate N negative samples')

    #training
    parser.add_argument('--bce_logits', type=int, default=0, help='use BCE with logits')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size for training')
    parser.add_argument('--n_epochs', type=int, default=10, help='Epoch number for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')

    config = parser.parse_args()

    train(config)