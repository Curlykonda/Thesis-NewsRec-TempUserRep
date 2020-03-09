import argparse
import os
import json
import gzip
import pickle
import time
import numpy as np
import pandas as pd

from collections import defaultdict

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

def get_art_id_from_dpg_history(articles_read):

    # format:
    # entry[0] = news_paper
    # entry[1] = article_id
    # entry[2] = time_stamp
    return [entry[1] for entry in articles_read]

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
