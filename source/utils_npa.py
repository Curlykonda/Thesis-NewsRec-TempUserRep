import argparse
import csv
import random
import nltk
from nltk.tokenize import word_tokenize
import datetime
import time
import random
import itertools
import numpy as np
import pickle
from numpy.linalg import cholesky

FILE_NAMES = {"click": "ClickData_sample.tsv", "news": "DocMeta_sample.tsv"}


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

def pad_sequence(seq, max_len, pad_value=0):
    if len(seq) < max_len:
        seq += [pad_value] * (max_len - len(seq))
    return seq

def preprocess_user_file(click_file='ClickData4.tsv', npratio=4, max_hist_len=50):
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

    all_train_id = []
    all_train_pn = []
    all_label = []

    all_test_id = []
    all_test_pn = []
    all_test_label = []
    all_test_index = []

    all_user_pos = []
    all_test_user_pos = []

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

        poslist = list(itertools.chain(*(trainpos))) # preserves browsing order of articles, last one at last pos in list
        neglist = list(itertools.chain(*(trainneg)))

        # extract test samples from the raw data
        if len(line) == 4:
            testimpre = [x.split('#TAB#') for x in line[3].split('#N#')]
            testpos = [x[0].split() for x in testimpre]
            testneg = [x[1].split() for x in testimpre]

            for i in range(len(testpos)):
                sess_index = []
                sess_index.append(len(all_test_pn))
                posset = list(set(poslist))

                # randomly select impressions from history
                allpos = [int(p) for p in random.sample(posset, min(max_hist_len, len(posset)))[:max_hist_len]]
                allpos += [0] * (max_hist_len - len(allpos))  # pad shorter sequences

                for j in testpos[i]:
                    all_test_pn.append(int(j))
                    all_test_label.append(1) # pos label := 1
                    all_test_id.append(userid_dict[userid])
                    all_test_user_pos.append(allpos)

                # add all neg test impression at this position to
                for j in testneg[i]:
                    all_test_pn.append(int(j))
                    all_test_label.append(0) # neg label := 0  (not clicked)
                    all_test_id.append(userid_dict[userid])
                    all_test_user_pos.append(allpos) # why add all positive instances here?

                sess_index.append(len(all_test_pn))
                # sess_index indicates start and end index of a session
                # length of session determined by the number of pos + neg test impressions
                all_test_index.append(sess_index)

        #HB check
        if len(poslist) != len(trainpos):
            # these collections can be of different length because we're unpacking the values into 'poslist'
            pass

        for impre_id in range(len(trainpos)):
            for pos_sample in trainpos[impre_id]:

                pos_neg_sample = newsample(trainneg[impre_id], npratio)
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
                allpos = [int(p) for p in random.sample(posset, min(max_hist_len, len(posset)))[:max_hist_len]] # sample random elems from set of pos impressions -> does order matter?
                allpos += [0] * (max_hist_len - len(allpos))
                all_train_pn.append(shuffle_sample) # ids
                all_label.append(shuffle_label)
                all_train_id.append(userid_dict[userid])
                all_user_pos.append(allpos)

                #if DEBUG
                if len(all_train_pn) % 1e3 == 0:
                    print("Train pn: {}".format(all_train_pn[-1]))
                    print("Labels: {}".format(all_label[-1]))
                    print("Pos: {}".format(all_user_pos[-1]))

    #reformat to np int arrays
    all_train_pn = np.array(all_train_pn, dtype='int32')
    all_label = np.array(all_label, dtype='int32')
    all_train_id = np.array(all_train_id, dtype='int32')
    all_test_pn = np.array(all_test_pn, dtype='int32')
    all_test_label = np.array(all_test_label, dtype='int32')
    all_test_id = np.array(all_test_id, dtype='int32')
    all_user_pos = np.array(all_user_pos, dtype='int32')
    all_test_user_pos = np.array(all_test_user_pos, dtype='int32')

    return userid_dict, all_train_pn, all_label, all_train_id, \
           all_test_pn, all_test_label, all_test_id, \
           all_user_pos, all_test_user_pos, all_test_index


def preprocess_news_file(news_file='DocMeta3.tsv'):
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
        if word_dict_raw[i][1] >= 2:
            word_dict[i] = [len(word_dict), word_dict_raw[i][1]]
    print("Vocab: {}  Raw: {}".format(len(word_dict), len(word_dict_raw)))

    news_words = [[0] * 30]  # max_len_news_title = 30
    news_index = {'0': 0}  # dictionary news indices
    for newsid in news:
        word_id = []
        news_index[newsid] = len(news_index)
        # get word_ids from news title
        for word in news[newsid][2]:
            if word in word_dict:
                word_id.append(word_dict[word][0])

        # pad sequence
        word_id = word_id[:30]  # max_len_news_title
        news_words.append(word_id + [0] * (30 - len(word_id)))  # note: 0 as padding value

    news_words = np.array(news_words, dtype='int32')
    return word_dict, news_words, news_index


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


def generate_batch_data_train(all_train_pn, all_label, all_train_id, batch_size, all_user_pos, news_words):
    inputid = np.arange(len(all_label))
    np.random.shuffle(inputid)
    y = all_label
    batches = [inputid[range(batch_size * i, min(len(y), batch_size * (i + 1)))] for i in
               range(len(y) // batch_size + 1)]

    print("test")

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


def main(config):

    userid_dict, all_train_pn, all_label, all_train_id, \
    all_test_pn, all_test_label, all_test_id, \
    all_user_pos, all_test_user_pos, all_test_index \
        = preprocess_user_file(click_file=config.data_path + FILE_NAMES["click"])

    word_dict, news_words, news_index = preprocess_news_file(news_file=config.data_path + FILE_NAMES["news"])

    traingen = generate_batch_data_train(all_train_pn, all_label, all_train_id, 100, all_user_pos, news_words)

    return traingen




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_path', type=str, default='../datasets/NPA/',
                        help='path to directory with NPA data')
    parser.add_argument('--word_emb_path', type=str, default='../embeddings/glove_eng.840B.300d.txt',
                        help='path to directory with word embeddings')

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
