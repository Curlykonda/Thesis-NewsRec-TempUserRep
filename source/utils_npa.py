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


def newsample(nnn, ratio):
    '''

    :param nnn: collection of elements from which to sample, e.g. article id for neg impressions
    :param ratio: total number of samples to be returned
    :return: random sample of size 'ratio' from the given collection 'nnn'
    '''
    if ratio > len(nnn):
        return random.sample(nnn * (ratio // len(nnn) + 1), ratio) # expand sequence with duplicates so that we can sample enough elems
    else:
        return random.sample(nnn, ratio)

def preprocess_npa_user_file(click_file='ClickData4.tsv', npratio=4, max_hist_len=50):
    '''

    :param click_file: path to file containing user click data
    :param npratio: ratio of negative samples, i.e. for each positive sample add N negative examples (randomly sampled articles)
    :param max_hist_len: maximum number of articles in user history; shorter sequences are padded with 0, longer ones are truncated
    :return:
    '''
    userid_dict = {}
    with open(click_file) as f:
        userdata = f.readlines()
    for user in userdata:
        line = user.strip().split('\t')
        userid = line[0]
        # map user_id to index
        if userid not in userid_dict:
            userid_dict[userid] = len(userid_dict)

    user_ids_train = []
    candidates_train = []
    labels_train = []
    user_hist_pos_train = []

    user_ids_test = []
    candidates_test = []
    labels_test = []
    indices_test = []
    user_hist_pos_test = []

    all_news_ids = set() # collect all occurring article ids

    for user in userdata:
        line = user.strip().split('\t')
        userid = line[0]
        # split userdata in impressions
        # format:
        # id #TAB# 'more ids seperated by spaces' #TAB# 'date' space 'time' #N# ...
        # probably #N# indicates new impression
        # line[2] is used for train samples
        # line[3] is used for test samples
        if len(line) == 4:
            impre = [x.split('#TAB#') for x in line[2].split('#N#')]
        if len(line) == 3:
            impre = [x.split('#TAB#') for x in line[2].split('#N#')]
        else:
            # raise NotImplementedError()  # HB edit
            pass

        # aggregate article ids of positive and neg impressions
        trainpos = [x[0].split() for x in impre]  # first element in that list has been clicked
        trainneg = [x[1].split() for x in impre]  # other elements have not been clicked, so use as 'neg' example

        # preserves browsing order of articles, last one at last pos in list
        poslist = [int(x) for x in list(itertools.chain(*(trainpos)))]
        neglist = [int(x) for x in list(itertools.chain(*(trainneg)))]

        # add news indices
        all_news_ids.update(poslist)

        # extract test samples from the raw data
        if len(line) == 4:
            testimpre = [x.split('#TAB#') for x in line[3].split('#N#')]
            testpos = [x[0].split() for x in testimpre]
            testneg = [x[1].split() for x in testimpre]

            all_news_ids.update(*testpos)
            all_news_ids.update(*testneg)

            for i in range(len(testpos)):
                sess_index = []
                sess_index.append(len(candidates_test))
                posset = list(set(poslist))

                # randomly select impressions from history
                allpos = [int(p) for p in random.sample(posset, min(max_hist_len, len(posset)))[:max_hist_len]]
                allpos += [0] * (max_hist_len - len(allpos))  # pad shorter sequences

                for j in testpos[i]:
                    candidates_test.append(int(j))
                    labels_test.append(1) # pos label := 1
                    user_ids_test.append(userid_dict[userid])
                    user_hist_pos_test.append(allpos)

                # add all neg test impression at this position to
                for j in testneg[i]:
                    candidates_test.append(int(j))
                    labels_test.append(0) # neg label := 0  (not clicked)
                    user_ids_test.append(userid_dict[userid])
                    user_hist_pos_test.append(allpos) # why add all positive instances here?

                sess_index.append(len(candidates_test))
                # sess_index indicates start and end index of a session
                # length of session determined by the number of pos + neg test impressions
                indices_test.append(sess_index)

        #HB check
        if len(poslist) != len(trainpos):
            # these collections can be of different length because we're unpacking the values into 'poslist'
            pass

        for impre_id in range(len(trainpos)):
            for pos_sample in trainpos[impre_id]:

                pos_neg_sample = newsample(trainneg[impre_id], npratio) # generate negative samples
                pos_neg_sample.append(pos_sample)

                temp_label = [0] * npratio + [1]
                temp_id = list(range(npratio + 1))
                random.shuffle(temp_id)

                shuffle_sample = []
                shuffle_label = []
                # shuffle article ids with corresponding label
                for id in temp_id:
                    shuffle_sample.append(int(pos_neg_sample[id]))
                    shuffle_label.append(temp_label[id])

                posset = list(set(poslist) - set([pos_sample])) # remove positive sample from user history

                # sample random elems from set of pos impressions -> does order matter?
                allpos = [int(p) for p in random.sample(posset, min(max_hist_len, len(posset)))[:max_hist_len]]
                allpos += [0] * (max_hist_len - len(allpos))
                candidates_train.append(shuffle_sample) # ids of candidate items
                labels_train.append(shuffle_label)
                user_ids_train.append(userid_dict[userid])
                user_hist_pos_train.append(allpos)

                # add news ids to grand set
                all_news_ids.update(shuffle_sample)
                #all_news_ids.update(allpos)

                #if DEBUG
                if len(candidates_train) % 1e3 == 0:
                    print("Train pn: {}".format(candidates_train[-1]))
                    print("Labels: {}".format(labels_train[-1]))
                    print("Pos: {}".format(user_hist_pos_train[-1]))

    #reformat to np int arrays

    candidates_train = np.array(candidates_train, dtype='int32')
    labels_train = np.array(labels_train, dtype='int32')
    user_ids_train = np.array(user_ids_train, dtype='int32')

    candidates_test = np.array(candidates_test, dtype='int32')
    labels_test = np.array(labels_test, dtype='int32')
    user_ids_test = np.array(user_ids_test, dtype='int32')
    user_hist_pos_train = np.array(user_hist_pos_train, dtype='int32')
    user_hist_pos_test = np.array(user_hist_pos_test, dtype='int32')

    # TODO: simplify the data format. put corresponding instacnes into one dict with a running index (e.g. train_pn, label, train_id, user_pos)
    columns = ["u_id", "history", "candidates", "labels"]
    train_data = [{'u_id': u_id, 'history': hist, 'candidates': cand, 'labels': lbl} for u_id, hist, cand, lbl
                    in zip(user_ids_train, user_hist_pos_train, candidates_train, labels_train)]

    test_data = [{'u_id': u_id, 'history': hist, 'candidates': cand, 'labels': lbl} for u_id, hist, cand, lbl
                    in zip(user_ids_test, user_hist_pos_test, candidates_test, labels_test)]

    return userid_dict, candidates_train, labels_train, user_ids_train, \
           candidates_test, labels_test, user_ids_test, \
           user_hist_pos_train, user_hist_pos_test, indices_test, all_news_ids

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
            candidate_articles = [art_id2idx[art_id] for art_id in newsample(all_article_ids, npratio)]
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

def preprocess_npa_news_file(news_file='DocMeta3.tsv', min_counts_for_vocab=2, max_len_news_title=30):
    with open(news_file) as f:
        newsdata = f.readlines()

    news = {}
    for newsline in newsdata:
        line = newsline.strip().split('\t')
        news[line[1]] = [line[2], line[3], word_tokenize(line[6].lower())]
    word_dict_raw = {'PADDING': [0, 999999]}  # key: 'word', value: [index, counts]

    for docid in news:
        for word in news[docid][2]:
            if word in word_dict_raw:
                word_dict_raw[word][1] += 1
            else:
                word_dict_raw[word] = [len(word_dict_raw), 1]
    word_dict = {}
    for i in word_dict_raw:
        # only include word with counter > 2
        if word_dict_raw[i][1] >= min_counts_for_vocab:
            word_dict[i] = [len(word_dict), word_dict_raw[i][1]]
    print("Vocab: {}  Raw: {}".format(len(word_dict), len(word_dict_raw)))

    news_words = [[0] * max_len_news_title]  # encoded news title
    news_index = {'0': 0}  # dictionary news indices
    for newsid in news:
        word_id = []
        news_index[newsid] = len(news_index)
        # get word_ids from news title
        for word in news[newsid][2]:
            # if word occurs in vocabulary, add the id
            # unknown words are omitted
            if word in word_dict:
                word_id.append(word_dict[word][0])

        # pad sequence
        word_id = word_id[:max_len_news_title]  # max_len_news_title
        news_words.append(word_id + [0] * (max_len_news_title - len(word_id)))  # note: 0 as padding value

    news_words = np.array(news_words, dtype='int32')

    return word_dict, news_words, news_index

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

def get_embedding(word_dict, emb_path):
    embedding_dict = {}

    # emb_path = '/data/wuch/glove.840B.300d.txt'

    # for each word in vocabulary, look up and store glove embedding vector
    # words from dictionary that don't appear in Glove are discarded (?)
    cnt = 0
    with open(emb_path, 'rb') as f:
        linenb = 0
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            line = line.split()
            word = line[0].decode()
            linenb += 1
            if len(word) != 0:
                vec = [float(x) for x in line[1:]]
                # only store words that appear in dictionary
                if word in word_dict:
                    embedding_dict[word] = vec
                    if cnt % 1000 == 0:
                        print(cnt, linenb, word)
                    cnt += 1

    embedding_matrix = [0] * len(word_dict)
    cand = []
    # convert embedding dict into matrix
    # pretrained values are floats
    #
    for i in embedding_dict:
        embedding_matrix[word_dict[i][0]] = np.array(embedding_dict[i], dtype='float32')
        cand.append(embedding_matrix[word_dict[i][0]])
    cand = np.array(cand, dtype='float32')
    mu = np.mean(cand, axis=0)
    Sigma = np.cov(cand.T)
    norm = np.random.multivariate_normal(mu, Sigma, 1)
    for i in range(len(embedding_matrix)):
        # if embedding is int, i.e. no pretrained value, initialise with values from normal distribution
        if type(embedding_matrix[i]) == int:
            embedding_matrix[i] = np.reshape(norm, 300)
    embedding_matrix[0] = np.zeros(300, dtype='float32')  # placeholder with zero values (?)
    embedding_matrix = np.array(embedding_matrix, dtype='float32')
    print("Shape Embedding matrix: {}".format(embedding_matrix.shape))

    return embedding_matrix

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

def generate_npa_batch_data_train(all_train_pn, all_label, all_train_id, batch_size, all_user_pos, news_words):
    inputid = np.arange(len(all_label))
    np.random.shuffle(inputid)
    y = all_label
    batches = [inputid[range(batch_size * i, min(len(y), batch_size * (i + 1)))] for i in
               range(len(y) // batch_size + 1)]

    while (True):
        for sample_indices in batches:
            '''
            What should happen here now? 
            1) Sample indices indicate which training samples to use from 'all_train_pn'
            2) Each entry in 'all_train_pn' contains candidate article ids that can complete the sequence, but only one is correct
            3) 'news_words' is a lookup containing the encoded news title for all articles. 
                In the provided sample data, however, we only have a small subset of all news articles (n=51), while the highest (listed) article_id is 42232.
            4) Split the 'candidates' into slices
            5) Get title encoding for all articles in reading histories (contained in 'all_user_pos')
            6) Split 'browsed_news' into slices
            7) Get 'user_ids'
            8) Get 'labels'  
            '''

            candidate = news_words[all_train_pn[sample_indices]] # shape: batch_size X n_candidates X title_len
            candidate_split = [candidate[:, k, :] for k in range(candidate.shape[1])] #candidate_split[0].shape := (batch_size, max_title_len)

            # retrieve reading histories for the sample indices
            browsed_news = news_words[all_user_pos[sample_indices]] # given article_ids for a user, retrieve the news title vector
            browsed_news_split = [browsed_news[:, k, :] for k in range(browsed_news.shape[1])]

            # get user_ids
            user_ids = np.expand_dims(all_train_id[sample_indices], axis=1) # expand dimensions to make 2D array
            label = all_label[sample_indices]

            #yield (candidate_split + browsed_news_split + [userid], label)
            return (candidate_split + browsed_news_split + [user_ids], label)

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


def main(config):

    if config.data_type not in DATA_TYPES:
        raise KeyError("{} is not a known data format".format(config.data_type))

    if config.data_type == "NPA":
        # use NPA sample data -> NOTE: cannot train model due to insufficient data!
        userid_dict, all_train_pn, all_label, all_train_id, \
        all_test_pn, all_test_label, all_test_id, \
        all_user_pos, all_test_user_pos, all_test_index, all_news_ids \
            = preprocess_npa_user_file(click_file=config.data_path + FILE_NAMES_NPA["click"])

        word_dict, news_words, news_index = preprocess_npa_news_file(news_file=config.data_path + FILE_NAMES_NPA["news"])

        traingen = generate_npa_batch_data_train(all_train_pn, all_label, all_train_id, 100, all_user_pos, news_words)

    elif config.data_type == "DPG":
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

        idx2u_id = reverse_mapping_dict(u_id2idx)
        idx2art_id = reverse_mapping_dict(art_id2idx)

        train_data, test_data = train_test_split(data, test_size=0.2, shuffle=True, random_state=42)

        batch = gen_batch_data(train_data, news_as_word_ids, batch_size=100)


    elif config.data_type == "Adressa":
        # use Adressa data
        raise NotImplementedError()

    return traingen



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
