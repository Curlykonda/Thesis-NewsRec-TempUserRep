import argparse
import os
import json
import gzip
import pickle
import time
import numpy as np
import pandas as pd
from collections import defaultdict

import torch

import sys
sys.path.append("..")

#import warnings
#warnings.filterwarnings('ignore')

#import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

import nltk
from nltk.tokenize import word_tokenize, sent_tokenize

#from transformers import BertTokenizer, BertModel
#import torch


def pad_sequence(seq, max_len, pad_value=0, pad='post', trunc='last'):
    if len(seq) < max_len:
        if pad == 'post':
            return seq + [pad_value] * (max_len - len(seq))
        elif pad == 'pre':
            return [pad_value] * (max_len - len(seq)) + seq
        else:
            raise NotImplementedError()

    elif len(seq) > max_len:
        if trunc == 'last':
            return seq[:max_len]
        elif trunc == 'first':
            return seq[len(seq) - max_len:]
        else:
            raise NotImplementedError()

    return seq

def get_art_id_from_dpg_history(articles_read, with_time=False):

    # format:
    # entry[0] = news_paper
    # entry[1] = article_id
    # entry[2] = time_stamp
    if with_time:
        return [(art_id, time_stamp) for _, art_id, time_stamp in articles_read]
    else:
        return [art_id for _, art_id, _ in articles_read]

def build_vocab_from_word_counts(vocab_raw, max_vocab_size, min_counts_for_vocab):
    vocab = {}
    # vocab = dict(heapq.nlargest(max_vocab_size, vocab.items(), key=lambda i: i[1]))
    for word, counter in vocab_raw.most_common(max_vocab_size):
        if counter >= min_counts_for_vocab:
            vocab[word] = len(vocab)
        else:
            break
    return vocab

def reverse_mapping_dict(item2idx):
    return {idx: item for item, idx in item2idx.items()}


def create_checkpoint(check_dir, filename, dataset, model, optimizer, results, step):

    checkpoint_path = check_dir / (f'{filename}_step_{step}.pt')

    print(f"Saving checkpoint to {checkpoint_path}")

    torch.save(
        {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'results': results,
            'dataset': dataset
        },
        checkpoint_path
    )

    print("Saved.")

def load_checkpoint(checkpoint_path, model, optimizer):
    #load checkpoint saved at checkpoint_path

    checkpoint = torch.load(checkpoint_path)
    dataset = checkpoint['dataset']
    step = checkpoint['step'] + 1
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    results = checkpoint['results']

    return dataset, results, step
