import numpy as np
import argparse
import pickle
import itertools
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
from source.utils_npa import get_dpg_data_processed, get_embeddings_from_pretrained
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

def test_eval_like_npa_wu(model, test_generator, act_func="sigmoid", one_candidate=False):
    '''
    Tests the model performance on an unseen test set. Method follows the functionality of the original NPA implementation.
    One testing instance should feature a mixed set of impressions, with positive and negative labels. The testing case described by the authors,
    these testing samples are obtained from a news aggregating platform and preprocessed by a separate function that has distinct features than for the training samples.
    Since we neither have access to this aggregating platform nor does our data format entail this information, the required test samples have to generated by a work-around function.

    :param model: trained NPA which to evaluate
    :param test_generator: Pytorch Dataloader that generates testing instances with corresponding labels
    :param one_candidate: parameter to indicate whether to use multiple or a single candidate
    :return:
        metrics_epoch : list where each entry contains the performance metrics for a single batch
        eval_str : string that reports performance on a small sub-sample
    '''
    # option to select a single candidate-target pair
    metrics_epoch = []
    model.eval()
    device = torch.device(model.device)

    click_scores = defaultdict(list)
    loss_epoch = []
    acc = []

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
            if len(logits.shape) < 2:  # cover border cases for batch size = 1
                logits = logits.unsqueeze(0)

            if act_func == "sigmoid":
                y_probs = torch.sigmoid(logits)
            else:
                y_probs = torch.nn.functional.softmax(logits, dim=1)
            y_probs_cpu = y_probs.detach().cpu()

            # compute loss
            test_loss = nn.BCELoss()(y_probs, lbls)
            #
            # aggregate predictions per user because each user can have various test samples
            for idx, u_id in enumerate(user_ids):
                u_id = u_id.item()

                preds = y_probs[idx].detach().cpu().numpy().tolist()
                targets = lbls[idx].cpu().numpy().tolist()
                if u_id not in click_scores:
                    click_scores[u_id] = []
                    click_scores[u_id].append(preds)
                    click_scores[u_id].append(targets)
                else:
                    click_scores[u_id][0].extend(preds)
                    click_scores[u_id][1].extend(targets)

            metrics_epoch.append(
                (test_loss.item(),
                compute_acc_tensors(y_probs, lbls),
                list(itertools.chain(*logits.detach().cpu().numpy()))
                 )
            )

            # precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
            if device.type == 'cpu':
                #break
                pass
    loss, acc, raw_scores = (zip(*metrics_epoch))

    metrics = defaultdict(list)
    for u_id in click_scores:
        preds = click_scores[u_id][0]
        targets = click_scores[u_id][1]
        metrics['auc'].append(roc_auc_score(targets, preds))
        metrics['ap'].append(average_precision_score(targets, preds))
        metrics['mrr'].append(mrr_score(targets, preds))
        metrics['ndcg5'].append(ndcg_score(targets, preds, k=5))
        metrics['ndcg10'].append(ndcg_score(targets, preds, k=10))

    metrics_epoch = [
        (np.mean(loss),
        np.mean(acc),
        np.mean(metrics['auc']),
        np.mean(metrics['ap']),
        np.mean(metrics['mrr']),
        np.mean(metrics['ndcg5']),
        np.mean(metrics['ndcg10']),
        list(itertools.chain(*raw_scores))
        )
    ]

    idx = min(5, lbls.shape[0]-1)
    if one_candidate:
        tgts = lbls[:idx].cpu().numpy().tolist()
    else:
        tgts = lbls[idx].cpu().numpy().tolist()
    eval_str = ("\nLogits {} \nProbs {} \nTargets {}".format(
        map_round_tensor(logits, idx=idx), map_round_tensor(y_probs, idx=idx), tgts))

    return metrics_epoch, eval_str

def train_npa_actfunc(npa_model, criterion, optim, train_generator, act_func="softmax"):
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
        if act_func == "sigmoid":
            y_probs = torch.sigmoid(logits)
        else:
            y_probs = torch.nn.functional.softmax(logits, dim=1)

        # compute loss
        loss_bce = criterion(y_probs, lbls)

        loss_l2 = None
        for param in npa_model.parameters():
            if loss_l2 is None:
                loss_l2 = param.norm(2)
            else:
                loss_l2 = loss_l2 + param.norm(2)

        loss_l2 = loss_l2 * config.lambda_l2
        loss_total = loss_bce + loss_l2

        # optimiser backward
        optim.zero_grad()
        loss_total.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        optim.step()

        # add metrics
        lbl_cpu = lbls.cpu()
        y_probs_cpu = y_probs.detach().cpu()


        mrr_scores, ndcg5_scores, ndcg10_scores = zip(*[(mrr_score(lbl, pred), ndcg_score(lbl, pred, k=5), ndcg_score(lbl, pred, k=10))
                                                        for lbl, pred in zip(lbl_cpu.numpy(), y_probs_cpu.detach().numpy())])

        metrics_epoch.append((loss_bce.item(),
                              compute_acc_tensors(y_probs_cpu, lbl_cpu),
                              roc_auc_score(lbl_cpu, y_probs_cpu),  # TPR v. FPR with varying threshold
                              average_precision_score(lbl_cpu, y_probs_cpu), # \text{AP} = \sum_n (R_n - R_{n-1}) P_n
                              mrr_scores,
                              ndcg5_scores,
                              ndcg10_scores,
                              list(itertools.chain(*logits.detach().cpu().numpy())),
                              loss_l2.item(),
                              loss_total.item()
                              ))

        if device.type == 'cpu': #and i_batch > 0: #
            print("Stopped after {} batches".format(i_batch + 1))
            break
    return metrics_epoch


def main(config):

    # set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    hyper_params = {'random_seed': config.random_seed,
                    'lr': config.lr, 'neg_sample_ratio': config.neg_sample_ratio, 'batch_size': config.batch_size,
                    'lambda_l2': config.lambda_l2, 'weight_decay': config.weight_decay,
                    'train_act_func': config.train_act_func, 'test_act_func': config.test_act_func,
                    'n_epochs': config.n_epochs, 'data_type': config.data_type
                    }

    #set random seeds
    torch.manual_seed(config.random_seed)
    torch.cuda.manual_seed(config.random_seed)
    np.random.seed(config.random_seed)

    # Get & prepare data
    # data is indexed by user_ids
    dataset, vocab, news_as_word_ids, art_id2idx, u_id2idx = get_dpg_data_processed(config.data_path, config.train_method, config.neg_sample_ratio,
                                                                                    config.max_hist_len, config.max_news_len, load_prepped=True)
    word_embeddings = get_embeddings_from_pretrained(vocab, emb_path=config.word_emb_path)

    train_data = dataset['train']
    test_data = dataset['test']

    # if use_val_set:
    #     train_data, val_data = train_test_split(train_data, test_size=0.1, shuffle=True, random_state=config.random_seed)

    train_dataset = DPG_Dataset(train_data, news_as_word_ids)
    train_generator = DataLoader(train_dataset, config.batch_size, shuffle=True)

    test_dataset = DPG_Dataset(test_data, news_as_word_ids)
    test_generator = DataLoader(test_dataset, config.batch_size, shuffle=False)

    print("Train on {} samples".format(train_dataset.__len__()))
    #
    # build model
    #
    model_params = {'n_users': len(dataset['train'])+len(dataset['test']), 'vocab_len': len(vocab),
                    'dim_user_id': 50, 'dim_pref_query': 200, 'dim_words': 300,
                    'max_title_len': config.max_hist_len, 'device': device,
                    'interest_extractor': config.interest_extractor}

    npa_model = NPA_wu(n_users=len(dataset['train'])+len(dataset['test']), vocab_len=len(vocab), pretrained_emb=word_embeddings,
                       emb_dim_user_id=50, emb_dim_pref_query=200, emb_dim_words=300,
                       max_news_len=config.max_news_len, max_hist_len=config.max_hist_len,
                       device=device, interest_extractor=config.interest_extractor)
    npa_model.to(device)
    #
    #npa_model.apply(init_weights)
    #
    #optim & loss
    criterion = nn.BCELoss() # raw_scores (logits) -> Softmax -> BCE loss
    if config.eval_method == 'wu':
        # create original NPA train and test setting
        optim = torch.optim.Adam(npa_model.parameters(), lr=0.001)
        config.train_act_func = "softmax"
        config.test_act_func = "sigmoid"
        config.test_w_one = False
    else:
        optim = torch.optim.Adam(npa_model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    #Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False) default

    print(device)
    print_setting(config, ['random_seed', 'train_method', 'eval_method', 'weight_decay', 'lambda_l2', 'train_act_func', 'test_act_func', 'data_path'])

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

    writer.add_hparams(hparam_dict=hyper_params, metric_dict={'test': 0.1})

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

        metrics_epoch = train_npa_actfunc(npa_model, criterion, optim, train_generator, act_func=config.train_act_func)
        t1 = time.time()
        metrics_train = log_metrics(epoch, metrics_epoch, metrics_train, writer, method=config.log_method)
        #
        ##########################################################################################
        #evaluate on test set
        ###############################
        if config.eval_method == 'wu':
            metrics_epoch, eval_msg = test_eval_like_npa_wu(npa_model, test_generator)
        elif config.eval_method == 'custom':
            metrics_epoch, eval_msg = test_eval_like_npa_wu(npa_model, test_generator, act_func=config.test_act_func,
                                                            one_candidate=config.test_w_one)
        else:
            raise KeyError("{} is no valid evluation method".format(config.eval_method))

        # logging
        t2 = time.time()
        metrics_test = log_metrics(epoch, metrics_epoch, metrics_test, writer, mode='test', method=config.log_method)

        print("\n {} epoch".format(epoch))
        print("TRAIN: BCE loss {:1.3f} \t acc {:0.3f} \t auc {:0.3f} \t ap {:0.3f} \t L2 loss: {:0.3f} in {:0.1f}s".format(
                metrics_train['loss'][-1], metrics_train['acc'][-1], metrics_train['auc'][-1], metrics_train['ap'][-1], metrics_train['loss_l2'][-1], (t1-t0)))
        print("TEST: BCE loss {:1.3f}  \t acc {:0.3f} \t auc {:0.3f} \t ap {:0.3f} in {:0.1f}s".format(
                metrics_test['loss'][-1], metrics_test['acc'][-1], metrics_test['auc'][-1], metrics_test['ap'][-1], (t2-t1)))

        print(eval_msg)

        if device.type == 'cpu':
            break

    print("\n---------- Done in {0:.2f} min ----------\n".format((time.time()-t_train_start)/60))
    #writer.add_figure()

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

    keys = ['loss', 'acc', 'auc', 'ap', 'mrr', 'ndcg5', 'ndcg10', 'loss_l2', 'loss_total']

    if mode =='train':
        loss, acc, auc, ap, mrr, ndcg5, ndcg10, logits, loss_l2, loss_total = (zip(*metrics_epoch))
        stats = (loss, acc, auc, ap, mrr, ndcg5, ndcg10, loss_l2, loss_total)
    else:
        loss, acc, auc, ap, mrr, ndcg5, ndcg10, logits = (zip(*metrics_epoch))
        stats = (loss, acc, auc, ap, mrr, ndcg5, ndcg10)

    for i, val in enumerate(stats):
        if method == 'epoch':
            metrics[keys[i]].append(np.mean(val))
            writer.add_scalar(keys[i] + '/' + mode, np.mean(val), global_step=epoch)
            writer.add_scalar(keys[i] + '-var/' + mode, np.var(val), global_step=epoch)

        elif method == 'batches' and mode != 'test':
            for v in val:
                writer.add_scalar(keys[i] + '/' + mode, v, len(metrics[keys[i]]))
                writer.add_scalar(keys[i] + '-var/' + mode, v, len(metrics[keys[i]]))

                metrics[keys[i]].append(v)

        else:
            raise NotImplementedError()

    logits = list(itertools.chain(*logits))
    writer.add_histogram("logits/" + mode, np.array(logits), epoch)
    # -> expect that logits widely distributed in the beginning but become more concentrated around certain high & low points

    #writer.close()
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #general
    parser.add_argument('--random_seed', type=int, default=42, help='random seed for reproducibility')

    # input data
    parser.add_argument('--data_type', type=str, default='DPG', help='options for data format: DPG, NPA or Adressa ')
    parser.add_argument('--data_path', type=str, default='../datasets/dpg/dev_time_split/', help='path to data directory') # dev : i10k_u5k_s30/
    parser.add_argument('--word_emb_path', type=str, default='../embeddings/glove_eng.840B.300d.txt',
                        help='path to directory with word embeddings')

    # preprocessing
    parser.add_argument('--max_hist_len', type=int, default=50,
                        help='maximum length of user reading history, shorter ones are padded; should be in accordance with the input datasets')
    parser.add_argument('--max_news_len', type=int, default=30,
                        help='maximum length of news article, shorter ones are padded; should be in accordance with the input datasets')
    parser.add_argument('--neg_sample_ratio', type=int, default=4,
                        help='Negative sample ratio N: for each positive impression generate N negative samples')
    parser.add_argument('--candidate_generation', type=str, default='neg_sampling',
                        help='Method to generate candidate articles: [neg_sampling, neg_sampling_time]')
    parser.add_argument('--train_method', type=str, default='pos_cut_off',
                        help='Method for network training & format of training samples: [wu, pos_cut_off, masked_interests]')

    #model params
    parser.add_argument('--interest_extractor', type=str, default=None,
                        help='[None, gru]')

    #training
    parser.add_argument('--batch_size', type=int, default=100, help='batch size for training')
    parser.add_argument('--n_epochs', type=int, default=10, help='Epoch number for training')
    parser.add_argument('--lambda_l2', type=float, default=0.0005, help='Parameter to control L2 loss')

    parser.add_argument('--train_act_func', type=str, default='softmax',
                        help='Output activation func Training: [softmax, sigmoid]')
    parser.add_argument('--test_act_func', type=str, default='sigmoid',
                        help='Output activation func Testing: [softmax, sigmoid]')
    parser.add_argument('--log_method', type=str, default='epoch', help='Mode for logging the metrics: [epoch, batches]')
    parser.add_argument('--test_w_one', type=bool, default=False, help='use only 1 candidate during testing')
    parser.add_argument('--eval_method', type=str, default='wu', help='Mode for evaluating NPA model: [wu, custom]')


    # optimiser
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='weight decay as regularisation option')

    #logging
    parser.add_argument('--results_path', type=str, default='../results/', help='path to save metrics')
    parser.add_argument('--exp_name', type=str, default='dev',
                        help='Addition to experiment name for logging, e.g. size of dataset [small, large]')

    config = parser.parse_args()

    main(config)