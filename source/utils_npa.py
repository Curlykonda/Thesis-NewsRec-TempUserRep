import argparse
import csv
import random
import nltk
from nltk.tokenize import word_tokenize

from collections import defaultdict, Counter

import datetime
import time
import random
import itertools
import numpy as np
import pickle

import fasttext

from source.utils import get_art_id_from_dpg_history, build_vocab_from_word_counts, pad_sequence, reverse_mapping_dict
from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer



DATA_TYPES = ["NPA", "DPG", "Adressa"]

FILE_NAMES_NPA = {"click": "ClickData_sample.tsv", "news": "DocMeta_sample.tsv"}


def sample_n_from_elements(elements, ratio):
    '''

    :param elements: collection of elements from which to sample, e.g. article id for neg impressions
    :param ratio: total number of samples to be returned
    :return: random sample of size 'ratio' from the given collection 'nnn'
    '''
    if ratio > len(elements):
        return random.sample(elements * (ratio // len(elements) + 1), ratio) # expand sequence with duplicates so that we can sample enough elems
    else:
        return random.sample(elements, ratio)

def prep_dpg_user_file(user_file, all_article_ids, art_id2idx, npratio=4, max_hist_len=50):

    with open(user_file, "rb") as fin:
        user_data = pickle.load(fin)

    u_id2idx = {}

    user_ids_train = []
    candidates_train = []
    labels_train = []
    user_hist_pos_train = []

    for u_id in user_data.keys():

        u_id2idx[u_id] = len(u_id2idx) # create mapping from u_id to index

        # aggregate article ids of positive
        pos_impre = get_art_id_from_dpg_history(user_data[u_id]["articles_read"])
        pos_impre = [art_id2idx[article] for article in pos_impre]

        for pos_sample in pos_impre:

            # generate negative samples
            candidate_articles = [art_id2idx[art_id] for art_id in sample_n_from_elements(all_article_ids, npratio)]
            candidate_articles.append(pos_sample)
            labels = [0] * npratio + [1] # create temp labels
            candidate_articles = list(zip(candidate_articles, labels)) # zip art_id and label
            random.shuffle(candidate_articles) # shuffle article ids with corresponding label
            candidate_articles = np.array(candidate_articles)

            pos_set = list(set(pos_impre) - set([pos_sample]))  # remove positive sample from user history

            # sample random elems from set of pos impressions
            hist = [int(p) for p in random.sample(pos_set, min(max_hist_len, len(pos_set)))[:max_hist_len]]
            hist += [0] * (max_hist_len - len(hist))

            candidates_train.append(candidate_articles[:, 0])  # ids of candidate items
            lbls = candidate_articles[:, 1]
            assert lbls.__contains__(1) # sanity check
            labels_train.append(lbls)

            user_ids_train.append(u_id2idx[u_id])
            user_hist_pos_train.append(hist)
    #
    #reformat to np int arrays

    candidates_train = np.array(candidates_train, dtype='int32')
    labels_train = np.array(labels_train, dtype='int32')
    user_ids_train = np.array(user_ids_train, dtype='int32')

    columns = ["u_id", "history", "candidates", "labels"]
    data = [{'u_id': u_id, 'history': hist, 'candidates': cand, 'labels': lbl} for u_id, hist, cand, lbl
                    in zip(user_ids_train, user_hist_pos_train, candidates_train, labels_train)]

    return u_id2idx, data

def preprocess_dpg_news_file(news_file, tokenizer, min_counts_for_vocab=2, max_len_news_title=30, max_vocab_size=30000):

    with open(news_file, 'rb') as f:
        news_data = pickle.load(f)

    vocab = defaultdict(int)
    news_as_word_ids = []
    art_id2idx = {}

    # 1. construct raw vocab
    #TODO: use existing libraries to build vocabulary
    print("construct raw vocabulary ...")
    vocab_raw = Counter({'PAD': 999999})

    for art_id in news_data:
        tokens = tokenizer(news_data[art_id]["snippet"].lower(), language='dutch')
        vocab_raw.update(tokens)
        news_data[art_id]['tokens'] = tokens

        if len(vocab_raw) % 1e4 == 0:
            print(len(vocab_raw))

    # 2. construct working vocab
    print("construct working vocabulary ...")
    vocab = build_vocab_from_word_counts(vocab_raw, max_vocab_size, min_counts_for_vocab)
    print("Vocab: {}  Raw: {}".format(len(vocab), len(vocab_raw)))
    #del(vocab_raw)

    # 3. encode news as sequence of word_ids
    print("encode news as word_ids ...")
    news_as_word_ids = [[0] * max_len_news_title]  # encoded news title
    art_id2idx = {'0': 0}  # dictionary news indices

    for art_id in news_data:
        word_ids = []

        art_id2idx[art_id] = len(art_id2idx) # map article id to index

        # get word_ids from news title
        for word in news_data[art_id]['tokens']:
            # if word occurs in vocabulary, add the id
            # unknown words are omitted
            if word in vocab:
                word_ids.append(vocab[word])

        # pad & truncate sequence
        news_as_word_ids.append(pad_sequence(word_ids, max_len_news_title))

    # reformat as array
    news_as_word_ids = np.array(news_as_word_ids, dtype='int32')

    return vocab, news_as_word_ids, art_id2idx

def get_embeddings_from_pretrained(vocab, emb_path, emb_dim=300):
    try:
        ft = fasttext.load_model(emb_path) # load pretrained vectors

        # check & adjust dimensionality
        if ft.get_dimension() != emb_dim:
            fasttext.util.reduce_model(ft, emb_dim)

        embedding_matrix = [0] * len(vocab)
        embedding_matrix[0] = np.zeros(emb_dim, dtype='float32')  # placeholder with zero values for 'PAD'

        for word, idx in vocab.items():
            embedding_matrix[idx] = ft[word] # how to deal with unknown words?

        embedding_matrix = np.array(embedding_matrix, dtype='float32')
        print("Shape Embedding matrix: {}".format(embedding_matrix.shape))

        return embedding_matrix

    except:
        print("Could not load word embeddings")

        return None

def generate_batch_data_test(all_test_pn, all_label, all_test_id, batch_size, all_test_user_pos, news_words):
    inputid = np.arange(len(all_label))
    y = all_label
    batches = [inputid[range(batch_size * i, min(len(y), batch_size * (i + 1)))] for i in
               range(len(y) // batch_size + 1)]

    while (True):
        for i in batches:
            candidate = news_words[all_test_pn[i]]
            browsed_news = news_words[all_test_user_pos[i]]
            browsed_news_split = [browsed_news[:, k, :] for k in range(browsed_news.shape[1])]
            userid = np.expand_dims(all_test_id[i], axis=1)
            label = all_label[i]

            yield ([candidate] + browsed_news_split + [userid], label)

def gen_batch_data(data, news_as_word_ids, batch_size=100, shuffle=True):

    #TODO: implement nice and more general dataloader
    n_batches = range(len(data) // batch_size + 1)

    if shuffle:
        random.shuffle(data)

    batches = [(batch_size * i, min(len(data), batch_size * (i + 1))) for i in
               n_batches]

    for start, stop in batches:
        # get data for this batch
        cands, hist, users, labels = zip(*[(news_as_word_ids[data_p['candidates']], news_as_word_ids[data_p['history']],
                                            data_p['u_id'], data_p['labels'])
                                                for data_p in data[start:stop]]) #return multiple lists from list comprehension

        # get candidates
        candidates = np.array(cands) # shape: batch_size X n_candidates X title_len
        candidates_split = [candidates[:, k, :] for k in range(candidates.shape[1])] # candidate_split[0].shape := (batch_size, max_title_len)

        # get history
        hist = np.array(hist)  # shape: batch_size X max_hist_len X max_title_len
        history_split = [hist[:, k, :] for k in range(hist.shape[1])]  # shape := (batch_size, max_title_len)

        # get user ids
        user_ids = np.expand_dims(np.array(users), axis=1)

        # get labels
        labels = np.array(labels)


        # aggregate to batch
        batch = (candidates_split + history_split + [user_ids], labels)

        return batch

def gen_batch_data_test(data, news_as_word_ids, batch_size=100, candidate_pos=0):
    n_batches = range(len(data) // batch_size + 1)

    batches = [(batch_size * i, min(len(data), batch_size * (i + 1))) for i in
               n_batches]

    for start, stop in batches:
        # get data for this batch
        cands, hist, users, labels = zip(*[(news_as_word_ids[data_p['candidates']], news_as_word_ids[data_p['history']],
                                            data_p['u_id'], data_p['labels'])
                                           for data_p in
                                           data[start:stop]])  # return multiple lists from list comprehension

        # get candidates
        candidates = np.array(cands)  # shape: batch_size X n_candidates X title_len
        candidate = candidates[:, candidate_pos, :]  # candidate.shape := (batch_size, max_title_len)

        # get history
        hist = np.array(hist)  # shape: batch_size X max_hist_len X max_title_len
        history_split = [hist[:, k, :] for k in range(hist.shape[1])]  # shape := (batch_size, max_title_len)

        # get user ids
        user_ids = np.expand_dims(np.array(users), axis=1)

        # get labels
        labels = np.array(labels)
        labels = labels[:, candidate_pos]

    yield ([candidate] + history_split + [user_ids], labels)

def get_labels_from_data(data):
    labels = {}
    for entry_dict in data: #columns = ["u_id", "history", "candidates", "labels"]
        labels[entry_dict['u_id']] = entry_dict['labels']
    return labels

def get_dpg_data(data_path, neg_sample_ratio=4, max_hist_len=50, max_news_len=30, min_counts_for_vocab=2, load_prepped=False):

    if load_prepped:
        with open(data_path + "news_prepped.pkl", 'rb') as fin:
            (vocab, news_as_word_ids, art_id2idx) = pickle.load(fin)
    else:

        path_article_data = data_path + "news_data.pkl"

        vocab, news_as_word_ids, art_id2idx = preprocess_dpg_news_file(news_file=path_article_data,
                                                                   tokenizer=word_tokenize,
                                                                   min_counts_for_vocab=min_counts_for_vocab,
                                                                   max_len_news_title=max_news_len)

        with open(data_path + "news_prepped.pkl", 'wb') as fout:
            pickle.dump((vocab, news_as_word_ids, art_id2idx), fout)

    path_user_data = data_path + "user_data.pkl"

    u_id2idx, data = prep_dpg_user_file(path_user_data, set(art_id2idx.keys()), art_id2idx,
                                        npratio=neg_sample_ratio, max_hist_len=max_hist_len)

    return data, get_labels_from_data(data), vocab, news_as_word_ids, art_id2idx, u_id2idx

def main(config):

    if config.data_type not in DATA_TYPES:
        raise KeyError("{} is not a known data format".format(config.data_type))

    if config.data_type == "DPG":
        # use DPG data

        # load & prep news data

        # TODO: add option to load existing data
        '''
        if config.load_prepped_news != None:
            try:
                pickle.load(new_file)
            except:
        '''
        vocab, news_as_word_ids, art_id2idx = preprocess_dpg_news_file(news_file=config.article_data,
                                                                       tokenizer=word_tokenize,
                                                                       min_counts_for_vocab=2,
                                                                       max_len_news_title=30)

        with open(config.data_path + "news_prepped.pkl", 'wb') as fout:
            pickle.dump((vocab, news_as_word_ids, art_id2idx), fout)

        u_id2idx, data = prep_dpg_user_file(config.user_data, set(art_id2idx.keys()), art_id2idx, npratio=config.neg_sample_ratio, max_hist_len=config.max_hist_len)

        #idx2u_id = reverse_mapping_dict(u_id2idx)
        #idx2art_id = reverse_mapping_dict(art_id2idx)

        train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True, random_state=42)

        batch = gen_batch_data(train_data, news_as_word_ids, batch_size=100)


    elif config.data_type == "Adressa":
        # use Adressa data
        raise NotImplementedError()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # input data
    parser.add_argument('--data_type', type=str, default='DPG',
                        help='options for data format: DPG, NPA or Adressa ')
    parser.add_argument('--data_path', type=str, default='../datasets/dpg/i10k_u5k_s30/',
                        help='path to data directory') #dpg/i10k_u5k_s30/

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

    config = parser.parse_args()

    main(config)
