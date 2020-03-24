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
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

import sys
sys.path.append("..")

from source.my_datasets import DPG_Dataset
from source.models.NPA import NPA_wu, init_weights
from source.utils_npa import get_dpg_data, get_embeddings_from_pretrained
from source.utils import print_setting, save_metrics_as_pickle, save_config_as_json, create_exp_name, save_exp_name_label
from source.metrics import *

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

def test_eval_npa_softmax(model, test_generator):

    # difference: select a single cand-target pair + sigmoid activation
    metrics_epoch = []
    model.eval()
    device = torch.device(model.device)

    with torch.no_grad():
        for sample in test_generator:
            user_ids, brows_hist, candidates = sample['input']
            lbls = sample['labels']
            lbls = lbls.float().to(device)
            # sub sample single candidate + label for inference

            # forward pass
            logits = model(user_ids.long().to(device), brows_hist.long().to(device),
                           candidates.long().to(device))

            y_probs = torch.nn.functional.softmax(logits, dim=1)
            y_probs_cpu = y_probs.detach().cpu()
            # compute loss
            test_loss = nn.BCELoss()(y_probs, lbls)

            metrics_epoch.append((test_loss.item(),
                                  compute_acc_tensors(y_probs_cpu, lbls.cpu()),
                                  roc_auc_score(lbls.cpu(), y_probs_cpu),
                                  average_precision_score(lbls.cpu(), y_probs_cpu)))

            # precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

            if device.type == 'cpu':
                break
    idx = 5
    tgts = lbls[idx].cpu().numpy().tolist()
    eval_str = ("\nLogits {} \nSigm p {} \nTargets {}".format(
        map_round_tensor(logits, idx=idx), map_round_tensor(y_probs, idx=idx), tgts))

    return metrics_epoch, eval_str

def test_eval_like_npa_wu(model, test_generator, one_candidate=True):

    # difference: select a single cand-target pair + sigmoid activation
    metrics_epoch = []
    model.eval()
    device = torch.device(model.device)

    with torch.no_grad():
        for sample in test_generator:
            user_ids, brows_hist, candidates = sample['input']
            lbls = sample['labels']
            lbls = lbls.float().to(device)
            # sub sample single candidate + label for inference
            if one_candidate:
                candidates = candidates[:, 1].unsqueeze(1)
                lbls = lbls[:, 1]

            # forward pass
            logits = model(user_ids.long().to(device), brows_hist.long().to(device),
                               candidates.long().to(device))

            y_probs_sigmoid = torch.sigmoid(logits)
            y_probs_cpu = y_probs_sigmoid.detach().cpu()
            # compute loss
            test_loss = nn.BCEWithLogitsLoss()(logits, lbls)

            metrics_epoch.append((test_loss.item(),
                                  compute_acc_tensors(y_probs_cpu, lbls.cpu()),
                                  roc_auc_score(lbls.cpu(), y_probs_cpu),
                                  average_precision_score(lbls.cpu(), y_probs_cpu)))

            # precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

            if device.type == 'cpu':
                break
    idx = 5
    if one_candidate:
        tgts = lbls[:idx].cpu().numpy().tolist()
    else:
        tgts = lbls[idx].cpu().numpy().tolist()
    eval_str = ("\nLogits {} \nSigm p {} \nTargets {}".format(
        map_round_tensor(logits, idx=idx), map_round_tensor(y_probs_sigmoid, idx=idx), tgts))

    return metrics_epoch, eval_str

def train_npa_wu_softmax(npa_model, criterion, optim, train_generator):
    metrics_epoch = []
    npa_model.train()
    device = torch.device(npa_model.device)

    for i_batch, sample in enumerate(train_generator):  # (hist_as_word_ids, cands_as_word_ids, u_id), labels
        npa_model.zero_grad()
        user_ids, brows_hist, candidates = sample['input']
        lbls = sample['labels']
        lbls = lbls.float().to(device)

        # forward pass
        logits = npa_model(user_ids.long().to(device), brows_hist.long().to(device), candidates.long().to(device))

        # activation function
        y_probs_softmax = torch.nn.functional.softmax(logits, dim=-1)

        # compute loss
        loss_bce = criterion(y_probs_softmax, lbls)

        # try_var_loss_funcs(logits, lbls, i_batch)

        # optimiser backward
        optim.zero_grad()
        loss_bce.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        optim.step()

        # add metrics
        lbl_cpu = lbls.cpu()
        y_probs_cpu = y_probs_softmax.detach().cpu()

        metrics_epoch.append((loss_bce.item(),
                              compute_acc_tensors(y_probs_cpu, lbl_cpu),
                              roc_auc_score(lbl_cpu, y_probs_cpu),  # TPR v. FPR with varying threshold
                              average_precision_score(lbl_cpu, y_probs_cpu)))  # \text{AP} = \sum_n (R_n - R_{n-1}) P_n

        if device.type == 'cpu': #and i_batch > 0
            print("Stopped after {} batches".format(i_batch + 1))
            break
    return metrics_epoch


def main(config):

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print_setting(config, ['random_seed', 'log_method', 'test_w_one', 'eval_method'])

    hyper_params = {'lr': None, 'neg_sample_ratio': None,
                    'batch_size': config.batch_size,
                     'random_seed': config.random_seed} # TODO: aggregate h_params for SummaryWriter

    #set random seeds
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    # Get & prepare data
    # data is indexed by user_ids
    dataset, vocab, news_as_word_ids, art_id2idx, u_id2idx = get_dpg_data(config.data_path, config.neg_sample_ratio,
                                                                          config.max_hist_len, config.max_news_len, load_prepped=True)
    word_embeddings = get_embeddings_from_pretrained(vocab, emb_path=config.word_emb_path)

    train_data, test_data = train_test_split(dataset, test_size=0.2, shuffle=True, random_state=config.random_seed)

    train_dataset = DPG_Dataset(train_data, news_as_word_ids)
    train_generator = DataLoader(train_dataset, config.batch_size)

    test_dataset = DPG_Dataset(test_data, news_as_word_ids)
    test_generator = DataLoader(test_dataset, config.batch_size)
    print("Train on {} samples".format(train_dataset.__len__()))
    #
    # build model
    #
    model_params = {'n_users': len(dataset), 'vocab_len': len(vocab),
                    'dim_user_id': 50, 'dim_pref_query': 200, 'dim_words': 300,
                    'max_title_len': config.max_hist_len, 'device': device}


    npa_model = NPA_wu(n_users=len(dataset), vocab_len=len(vocab), pretrained_emb=word_embeddings,
                       emb_dim_user_id=50, emb_dim_pref_query=200, emb_dim_words=300,
                       max_news_len=config.max_news_len, device=device)
    npa_model.to(device)
    #
    #npa_model.apply(init_weights)
    #
    #optim & loss
    criterion = nn.BCELoss() # raw_scores (logits) -> Softmax -> BCE loss
    if config.eval_method == 'wu':
        #optim = torch.optim.Adam(npa_model.parameters(), lr=config.lr)
        optim = torch.optim.Adam(npa_model.parameters(), lr=config.lr, weight_decay=0.9)
    else:
        optim = torch.optim.Adam(npa_model.parameters(), lr=config.lr, weight_decay=0.9)
    #Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False) default

    # create dir for logging
    now = datetime.now()
    date = now.strftime("%m-%d-%y")

    res_path = Path(config.results_path)
    res_path = res_path / date
    try:
        n_exp = len(os.listdir(res_path)) + 1
    except:
        n_exp = 1

    exp_name = create_exp_name(config, n_exp, time=now.strftime("%H:%M"))
    res_path = res_path / exp_name
    res_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(res_path) # logging

    metrics_train = defaultdict(list)
    metrics_test = defaultdict(list)

    print_shapes = False
    DEBUG = True
    #
    ##########################
    # training
    t_train_start = time.time()
    for epoch in range(config.n_epochs):
        t0 = time.time()
        npa_model.train()

        metrics_epoch = train_npa_wu_softmax(npa_model, criterion, optim, train_generator)

        #loss, acc, auc = zip(*metrics)
        t1 = time.time()
        metrics_train = log_metrics(epoch, metrics_epoch, metrics_train, writer, method=config.log_method)
        #
        #####################
        #
        #evaluate on test set
        if config.eval_method == 'wu':
            metrics_epoch, eval_msg = test_eval_like_npa_wu(npa_model, test_generator, one_candidate=config.test_w_one)
        elif config.eval_method == 'softmax':
            metrics_epoch, eval_msg = test_eval_npa_softmax(npa_model, test_generator)
        else:
            raise KeyError("{} is no valid evluation method".format(config.eval_method))

        # logging
        t2 = time.time()
        metrics_test = log_metrics(epoch, metrics_epoch, metrics_test, writer, mode='test', method=config.log_method)

        print("\n {} epoch".format(epoch))
        print("TRAIN: BCE loss {:1.3f} \t acc {:0.3f} \t auc {:0.3f} \t ap {:0.3f} in {:0.1f}s".format(
                metrics_train['loss'][-1], metrics_train['acc'][-1], metrics_train['auc'][-1], metrics_train['ap'][-1], (t1-t0)))
        print("TEST: BCE loss {:1.3f}  \t acc {:0.3f} \t auc {:0.3f} \t ap {:0.3f} in {:0.1f}s".format(
                metrics_test['loss'][-1], metrics_test['acc'][-1], metrics_test['auc'][-1], metrics_test['ap'][-1], (t2-t1)))

        print(eval_msg)

        if device.type == 'cpu':
            break

    print("\n---------- Done in {0:.2f} min ----------\n".format((time.time()-t_train_start)/60))
    #writer.add_figure()
    #write.add_hparams
    writer.close()

    #save metrics & config
    metrics = {key: (metrics_train[key], metrics_test[key]) for key in metrics_test.keys()}
    save_metrics_as_pickle(metrics, res_path, file_name='metrics_' + config.log_method)
    save_config_as_json(config, res_path)
    save_exp_name_label(config, res_path, exp_name)


def map_round_tensor(tensor, decimals=3, idx=0):
    if len(tensor.shape) > 1:
        if idx > tensor.shape[0]:
            idx = -1
        return list(map(lambda x: x.round(decimals), tensor[idx].detach().cpu().numpy()))
    else:
        return list(map(lambda x: x.round(decimals), tensor[:idx].detach().cpu().numpy()))

def log_metrics(epoch, metrics_epoch, metrics, writer, mode='train', method='epoch'):
    loss, acc, auc, ap = (zip(*metrics_epoch))
    stats = {'loss': loss,
            'acc': acc,
            'auc': auc,
            'ap': ap
    }

    for key, val in stats.items():
        if method == 'epoch':
            metrics[key].append(np.mean(val))
            writer.add_scalar(key + '/' + mode, np.mean(val), epoch)
            writer.add_scalar(key + '-var/' + mode, np.var(val), epoch)

        elif method == 'batches':
            for v in val:
                writer.add_scalar(key + '/' + mode, v, len(metrics[key]))
                writer.add_scalar(key + '-var/' + mode, v, len(metrics[key]))

                metrics[key].append(v)

        else:
            raise NotImplementedError()

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #general
    parser.add_argument('--random_seed', type=int, default=42,
                        help='random seed for reproducibility')

    # input data
    parser.add_argument('--data_type', type=str, default='DPG',
                        help='options for data format: DPG, NPA or Adressa ')
    parser.add_argument('--data_path', type=str, default='../datasets/dpg/dev/',
                        help='path to data directory') # dev : i10k_u5k_s30/

    parser.add_argument('--word_emb_path', type=str, default='../embeddings/glove_eng.840B.300d.txt',
                        help='path to directory with word embeddings')

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
    parser.add_argument('--log_method', type=str, default='epoch', help='Mode for logging the metrics: [epoch, batches]')
    parser.add_argument('--test_w_one', type=bool, default=True, help='use only 1 candidate during testing')
    parser.add_argument('--eval_method', type=str, default='wu', help='Mode for evaluating NPA model: [wu, softmax]')

    # optimiser
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay as regularisation option')

    #logging
    parser.add_argument('--results_path', type=str, default='../results/', help='path to save metrics')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Addition to experiment name for logging, e.g. size of dataset [small, large]')

    config = parser.parse_args()

    main(config)