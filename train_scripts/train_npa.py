import numpy as np
import argparse
import pickle
import os
from pathlib import Path
import time
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
from source.metrics import *

import sys
sys.path.append("..")

from source.my_datasets import DPG_Dataset
from source.models.NPA import NPA_wu
from source.utils_npa import get_dpg_data, get_embeddings_from_pretrained

import source.utils_npa
import source.utils

def try_var_loss_funcs(logits, targets, i_batch):

    softmax = nn.Softmax(dim=1)
    sigmoid = nn.Sigmoid()
    log_softm = nn.LogSoftmax(dim=1)

    bce = nn.BCELoss()
    bce_w_log = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    nll = nn.NLLLoss()
    acc = compute_acc_tensors(logits, targets)

    softm_probs = softmax(logits)
    sigm_probs = sigmoid(logits)
    log_softm_probs = log_softm(logits)

    print("\n Batch {}".format(i_batch))
    print("TP acc {0:.3f}".format(acc))
    print("BCE softm {0:.3f} \t sigmoid {0:.3f}".format(bce(softm_probs, targets), bce(sigm_probs, targets)))
    print("BCE w Logits {0:.3f} \t softm {0:.3f}".format(bce_w_log(logits, targets), bce_w_log(softm_probs, targets)))
    print("CE softm {0:.3f} \t sigmoid {0:.3f}".format(ce(softm_probs, targets.argmax(dim=1)), ce(log_softm_probs, targets.argmax(dim=1))))
    print("NLL softm {0:.3f} \t log softm {0:.3f}".format(nll(softm_probs, targets.argmax(dim=1)), nll(log_softm_probs, targets.argmax(dim=1))))


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
    dataset, vocab, news_as_word_ids, art_id2idx, u_id2idx = get_dpg_data(config.data_path, config.neg_sample_ratio,
                                                                          config.max_hist_len, config.max_news_len)
    word_embeddings = get_embeddings_from_pretrained(vocab, emb_path=config.word_emb_path)

    train_data, test_data = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=config.random_seed)

    train_dataset = DPG_Dataset(train_data, news_as_word_ids)
    train_generator = DataLoader(train_dataset, config.batch_size)

    test_dataset = DPG_Dataset(test_data, news_as_word_ids)
    test_generator = DataLoader(test_dataset, config.batch_size)
    print("Train on {} samples".format(train_dataset.__len__()))
    #
    # build model
    npa_model = NPA_wu(n_users=len(dataset), vocab_len=len(vocab), pretrained_emb=word_embeddings,
                       emb_dim_user_id=50, emb_dim_pref_query=200, emb_dim_words=300, max_title_len=config.max_hist_len)
    npa_model.to(device)
    #
    #optim & loss
    if config.bce_logits:
        #crit_bce_logits = nn.BCEWithLogitsLoss()
        criterion = nn.BCEWithLogitsLoss() # raw_score -> nn.Sigmoid() -> BCE loss
        print("using BCE with Logits")
    else:
        #crit_bce = nn.BCELoss()
        criterion = nn.BCELoss() # raw_scores (logits) -> Softmax -> BCE loss
        # logits are un-normalized scores
    optim = torch.optim.Adam(npa_model.parameters(), lr=0.001)


    now = datetime.now()
    date = now.strftime("%m-%d-%y")
    res_path = Path(config.results_path)
    res_path = res_path / date
    res_path.mkdir(parents=True, exist_ok=True)

    exp_name = now.strftime("%H:%M") + '-metrics.pkl'
    writer = SummaryWriter(res_path) # logging

    metrics = defaultdict(list)
    print_shapes = False
    DEBUG = False

    t_train_start = time.time()
    for epoch in range(config.n_epochs):
        t0 = time.time()
        npa_model.train()

        metrics_epoch = []

        for i_batch, sample in enumerate(train_generator):  # (hist_as_word_ids, cands_as_word_ids, u_id), labels
            npa_model.zero_grad()
            user_ids, brows_hist, candidates = sample['input']
            lbls = sample['labels']
            lbls = lbls.float().to(device)

            # forward pass
            logits = npa_model(user_ids.long().to(device), brows_hist.long().to(device), candidates.long().to(device))
            if print_shapes:
                npa_model.get_representation_shapes()
                print_shapes = False

            y_probs = torch.nn.functional.softmax(logits, dim=1)
            y_preds = y_probs.detach().cpu().argmax(axis=1)

            # compute loss
            if config.bce_logits:
                loss_bce = criterion(logits, lbls) #loss_bce_logits, i.e. raw click scores
            else:
                loss_bce = criterion(y_probs, lbls) # use softmax probabilities

            try_var_loss_funcs(logits, lbls, i_batch)


            # optimiser backward
            optim.zero_grad()
            loss_bce.backward()
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
            optim.step()

            # add metrics
            #accuracy = (predictions.argmax(dim=1) == batch_targets).sum().float() / (config.batch_size)

            metrics_epoch.append((loss_bce.item(),
                                  compute_acc_tensors(y_probs, lbls),
                                  roc_auc_score(lbls.cpu(), y_probs.detach().cpu().numpy())))

            if DEBUG:
                break

        # metrics = logging(metrics_epoch, metrics, writer, mode='train')
        #loss, acc, auc = zip(*metrics)
        t1 = time.time()
        metrics = log_metrics(epoch, metrics_epoch, metrics, writer)

        #evaluate on test set
        metrics_epoch = []
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

            metrics_epoch.append((test_loss.item(),
                                  compute_acc_tensors(y_probs, lbls),
                                  roc_auc_score(lbls.cpu(), y_probs.detach().cpu().numpy())))

            if DEBUG:
                break

        # logging
        t2 = time.time()
        metrics = log_metrics(epoch, metrics_epoch, metrics, writer, mode='test')

        print("\n {} epoch".format(epoch))
        print("TRAIN: BCE loss {:1.3f} \t acc {:0.3f} in {:0.1f}s".format(metrics['loss/train'][-1], metrics['acc/train'][-1], (t1-t0)))
        print("TEST: BCE loss {:1.3f}  \t acc {:0.3f} in {:0.1f}s".format(metrics['loss/test'][-1], metrics['acc/test'][-1], (t2-t1)))

        if DEBUG:
            break

    print("\n---------- Done in {0:.2f} min ----------".format((time.time()-t_train_start)/60))
    #writer.add_figure()
    #write.add_hparams

    #save metrics
    with open(res_path / exp_name, 'wb') as fout:
        pickle.dump(metrics, fout)


def log_metrics(epoch, metrics_epoch, metrics, writer, mode='train'):
    loss, acc, auc = (zip(*metrics_epoch))

    metrics['loss/' + mode].append(np.mean(loss))
    metrics['acc/' + mode].append(np.mean(acc))
    metrics['auc/' + mode].append(np.mean(auc))
    writer.add_scalar('loss/' + mode, np.mean(loss), epoch)
    writer.add_scalar('acc/' + mode, np.mean(acc), epoch)
    writer.add_scalar('auc/' + mode, np.mean(auc), epoch)
    writer.add_scalar('loss-var/' + mode, np.var(loss), epoch)
    writer.add_scalar('acc-var/' + mode, np.var(acc), epoch)
    writer.add_scalar('auc-var/' + mode, np.var(auc), epoch)

    return metrics


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

    parser.add_argument('--results_path', type=str, default='../results/exp1/', help='path to save metrics')

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