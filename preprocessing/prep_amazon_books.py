from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import json
import gzip
import pickle
import time
import numpy as np
import pandas as pd

import sys
sys.path.append("..")

#import warnings
#warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

import nltk
#nltk.download('poplular')
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

#from pytorch_pretrained_bert import BertTokenizer, BertModel
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import DataLoader, Dataset


def parse2(f):
    for l in f:
        yield json.loads(l.strip())


def clean_text(sent, tokenizer, stop_words=None, stemmer=None):

    # tokenise & punctuation
    tokens = [word for word in tokenizer.tokenize(sent) if word.isalpha()]

    if not stop_words == None:
        #stop_words = list(set(stopwords.words('english')))
        # filter stop words
        tokens = [word for word in tokens if word not in stop_words]

    if not stemmer == None:
        tokens = [stemmer.stem(token) for token in tokens]

    # locating and correcting common typos & misspellings

    return tokens

def truncate_seq(tokens, tokenizer, max_len=512):
    if len(tokens) < max_len:
        tokens=tokens[:-1]
        n = max_len - len(tokens) - 1
        tokens += n * [tokenizer.pad_token]
    elif len(tokens) > max_len:
        tokens=tokens[:max_len-1]
    else:
        return tokens
    
    tokens.append(tokenizer.sep_token)
    
    return tokens

def tokenize_text_to_ids(text, tokenizer, sent_tokenize, max_len=512):
    sents = sent_tokenize(text)
    tokens = []
    n_words = 0

    tokens.append(tokenizer.cls_token)
    added_tokens = 1

    for s in sents:
        # tokenise & punctuation
        #tokens.extend(tokenizer.tokenize(s))
        tokens.extend(tokenizer.tokenize([word for word in tokenizer.tokenize(s) if word.isalpha()]))
        tokens.append(tokenizer.sep_token)
        added_tokens += 1
    
    n_words = len(tokens) - added_tokens
    tokens = truncate_seq(tokens, tokenizer, max_len)
    
    assert len(tokens) == max_len
    
    return tokenizer.convert_tokens_to_ids(tokens), n_words

def clean_df(df):
    ### remove rows with unformatted title (i.e. some 'title' may still contain html style content)

    df = df.fillna('')
    print(df.columns)
    df = df.drop(['reviewerName', 'style', 'vote'], axis=1)
    #df = df.drop('reviewerName', axis=1)
    #df = df.drop('style', axis=1)
    #df = df.drop('vote', axis=1)
    #df = df.drop('image', axis=1)

    # reformat timestamp
    df['reviewTime'] = df['unixReviewTime'].apply(lambda x: pd.Timestamp(x, unit='s'))
    df = df.drop('unixReviewTime', axis=1)

    # rename columns
    df = df.rename(columns={'asin': 'itemID'})

    # aggregate text
    df['reviewText'] = df['summary'] + " " + df['reviewText']
    df = df.drop('summary', axis=1)

    ## convert to other (smaller) datatype
    df['overall'] = df['overall'].astype('int32')

    return df

def prep_text(df, bert_tokenizer, bert_model, batch_size, max_len, device, drop_org_reviews=False):

    print("Encoding Text..")

    bert_model.to(device)

    encoded_text = {}
    start_idx = 0
    stop_idx = batch_size

    df["n_words"] = 0

    while(start_idx < len(df)):

        if stop_idx > len(df):
            stop_idx = len(df)

        encoded_in, n_words = zip(*[tokenize_text_to_ids(text, bert_tokenizer, sent_tokenize, max_len) for text
                  in list(df['reviewText'])[start_idx:stop_idx]])
        
        encoded_in = torch.tensor(encoded_in, requires_grad=False, device=device).long()

        with torch.no_grad():
            last_hidden, pooled_out = bert_model(encoded_in)

        assert last_hidden.shape[0] == batch_size

        #save tensor of encoded text into separate dictionary
        keys = range(start_idx, stop_idx)
        
        encoded_text = {**encoded_text, **dict(zip(keys, last_hidden))}

        df.loc[keys, "n_words"] = n_words

        start_idx += (batch_size+1)
        stop_idx += (batch_size+1)

    print("done")
    if drop_org_reviews:
        df = df.drop('reviewText', axis=1)

    df.astype({"n_words":'int32'})
        
    return df, encoded_text

def aggregate_info(df, user_d, items_d):
    added = 0
    updated = 0

    print("Aggregating User Stats..")
    for u, group in tqdm(df.groupby('reviewerID')):

        if u not in user_d:
            user_d[u] = {}
            user_d[u]['n_reviews'] = group['itemID'].count()
            user_d[u]['m_rating'] = group['overall'].mean()
            user_d[u]['m_words'] = group['reviewWords'].mean()
            added += 1
        else:
            user_d[u]['n_reviews'] += group['itemID'].count()
            user_d[u]['m_rating'] = np.mean([user_d[u]['m_rating'], group['overall'].mean()])
            user_d[u]['m_words'] = np.mean([user_d[u]['m_words'], group['reviewWords'].mean()])
            updated += 1

    print("Total users: %.0f" % len(user_d.keys()))
    print("Added: %.0f" % added)
    print("Updated: %.0f" % updated)

    # count reviews per item
    added = 0
    updated = 0

    print("Aggregating Item Stats..")
    for item, group in tqdm(df.groupby('itemID')):

        if item not in items_d:
            items_d[item] = {}
            items_d[item]['n_reviews'] = group['reviewerID'].count()
            items_d[item]['m_rating'] = group['overall'].mean()
            items_d[item]['m_words'] = group['reviewWords'].mean()
            added += 1
        else:
            items_d[item]['n_reviews'] += group['reviewerID'].count()
            items_d[item]['m_rating'] = np.mean([items_d[item]['m_rating'], group['overall'].mean()])
            items_d[item]['m_words'] = np.mean([items_d[item]['m_words'], group['reviewWords'].mean()])
            updated += 1

    print("Total items: %.0f" % len(items_d.keys()))
    print("Added: %.0f" % added)
    print("Updated: %.0f" % updated)

    return user_d, items_d

def print_time(t0):
    print("Done after in {}".format(time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))))

def preprocessDF_lisa(path, pkl_path, bert_tokenizer, bert_model, device, batch_size, max_len=200, drop_org_reviews=False):
    
    #init variables
    user_dict = {}
    item_dict = {}

    if not config.load_pkl == None:
        print("Loading intermediate results from {}".format(config.load_pkl))

        with open(config.load_pkl, 'rb') as fin: 
            df = pickle.load(fin)

        print("Lines in DF: {}".format(len(df)))

    else:
        i, j = 0, 0
        data = {}
        t1 = time.time()

        
        with gzip.open(path, 'rb') as f:
            print("Start reading: {}".format(path))
            for d in parse2(f):
                data[i] = d
                
                if not (i % 1e6):
                    print(j)
                    
                    #parser early stop
                    if j == config.n_reviews:
                        break
                    else:
                        j+=1
                i+=1

        print("Lines read: {} in {}".format(i+1, time.strftime("%H:%M:%S", time.gmtime(time.time() - t1))))

        print("Constructing DataFrame..")
        t1 = time.time()
        df = pd.DataFrame.from_dict(data, orient='index')
        print_time(t1)

        # preprocessing
        print("Cleaning DF..")
        t1 = time.time()
        df = clean_df(df)
        print_time(t1)

        #save intermediate df
        print("Saving intermediate..")
        df.to_pickle(pkl_path + "df.pkl")
        print("Pickle Protocol: {}".format(pickle.HIGHEST_PROTOCOL))

    #
    #t1 = time.time()
    #print("Cleaning Text..")
    #df['reviewWords'] = df['reviewText'].apply(lambda x: len(clean_text(x, bert_tokenizer)))
    #print_time(t1)

    # encode review text with Bert
    t1 = time.time()
    df, encoded_text = prep_text(df, bert_tokenizer, bert_model, batch_size, max_len=max_len, drop_org_reviews=drop_org_reviews, device=device)
    print_time(t1)

    # aggregate info
    user_dict, item_dict = aggregate_info(df, user_dict, item_dict)

    # save cleaned dataframe & encoded text
    print("Saving ..")
    df.to_pickle(pkl_path + "df.pkl")
    with open(pkl_path + "encoded_text.pkl", 'wb') as fout:
        pickle.dump(encoded_text, fout)

    # save stats
    with open(pkl_path + "user_stats.pkl", 'wb') as fout:
        pickle.dump(user_dict, fout)
    with open(pkl_path + "item_stats.pkl", 'wb') as fout:
        pickle.dump(item_dict, fout)
    # also with pickle protocol 4
    with open(pkl_path + "user_stats4.pkl", 'wb') as fout:
        pickle.dump(user_dict, fout, protocol=4)
    with open(pkl_path + "item_stats4.pkl", 'wb') as fout:
        pickle.dump(item_dict, fout, protocol=4)

    print("\n")
    print("DONE")

    return user_dict, item_dict


def print_flags(config):
    """
  Prints all entries in config variable.
  """
    for key, value in vars(config).items():
        print(key + ' : ' + str(value))

def main(config):
    print_flags(config)

    #set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("Device: {}" .format(device))

    print("Pickle Protocol: {}".format(pickle.HIGHEST_PROTOCOL))

    #load BERT
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_tokenizer.add_special_tokens({"unk_token": '[UNK]', 'cls_token': '[CLS]',
                                       'pad_token': '[PAD]', 'sep_token': '[SEP]'})
    #print(len(bert_tokenizer))
    assert bert_tokenizer.cls_token == '[CLS]'

    bert_model = BertModel.from_pretrained('bert-base-uncased')
    bert_model.resize_token_embeddings(len(bert_tokenizer))

    data_path = config.dataset
    pkl_path = config.pkl_path

    user_dict = {}
    item_dict = {}

    user_dict, item_dict = preprocessDF_lisa(data_path, pkl_path, bert_tokenizer, bert_model,
                                             device=device, batch_size=config.batch_size, max_len=config.max_seq_length, drop_org_reviews=config.drop_org_reviews)



if __name__ == "__main__":
    print("Python code starts now!")

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='../datasets/Books_5.json.gz',
                        help='path to the dataset file of format *.json.gz')
    parser.add_argument('--pkl_path', type=str, default='../datasets/books-pickle/', help='path to save pickle files')
    parser.add_argument('--batch_size', type=int, default=128, help='number of review in one batch for Bert')
    parser.add_argument('--max_seq_length', type=int, default=512,
                        help='maximum length of review, shorter ones are padded; should be in accordance with the input datasets')
    parser.add_argument('--drop_org_reviews', type=int, default=0,
                        help='After encoding the original review text is discarded to save space')
    parser.add_argument('--n_reviews', type=int, default=-1,
                        help='Number of reviews parsed; n * 1 Million lines; -1 reads all')
    parser.add_argument('--load_pkl', type=str, default=None,
                        help='path to pickle file with intermediate results')


    config = parser.parse_args()

    main(config)


