import argparse
import json
import gzip
import os
import pickle
from pathlib import Path
import smart_open
import random

import datetime

from collections import OrderedDict, defaultdict, Counter
from tqdm import tqdm

import sys
sys.path.append("..")

DATA_SIZES = {'dev': [10e3, 2e3], 'medium': [45e3, 10e3], 'large': [250e3, 50e3]}


def data_stream_generator(data_dir):
    files = [file for file in Path(data_dir).iterdir() if
             file.is_file() and file.name[0] not in '_.']
    for file in files:
        with smart_open.open(file) as rf:
            yield from (json.loads(line) for line in rf.read().splitlines() if line)

def get_text_snippet(text, len_snippet, tokenizer=None):
    if tokenizer is None:
        naive_tokens = str(text).split(" ")[:len_snippet]
        if len(naive_tokens) > len_snippet:
            tokens = naive_tokens[:len_snippet]
        else:
            tokens = naive_tokens
    else:
        #tokens = tokenizer.tokenize(text)
        raise NotImplementedError()

    return " ".join(tokens)

def get_n_words(text, tokenizer=None):
    if tokenizer is None:
        tokens = str(text).split(" ")
    else:
        # tokens = tokenizer.tokenize(text)
        raise NotImplementedError()

    return len(tokens)


# def subsample_items(n_items, snippet_len, data_dir, keys_to_exclude = ["short_id", "url"], test_time_thresh=None):
#     item_dict = OrderedDict()
#     item_dict['all'] = {}
#
#     # if test_time_thresh is not None:
#     #     item_dict['train'] = {}
#     #     item_dict['test'] = {}
#
#     for i, item in enumerate(data_stream_generator(data_dir + "items")):
#         #item.keys() = dict_keys(['text', 'pub_date', 'author', 'url', 'short_id'])
#         if snippet_len is not None:
#             item['snippet'] = get_text_snippet(item['text'], snippet_len)
#
#         item['n_words'] = get_n_words(item['text'])
#         val_dict = {key: val for (key, val) in item.items() if key not in keys_to_exclude}
#         item_dict['all'][item['short_id']] = val_dict
#
#         if item['pub_date']: #< test_time_thresh:
#              pass # currently no pub_date available [30.03.]
#
#         if i == n_items-1:
#             break
#
#     return item_dict

def update_logging_dates(articles_read, logging_dates):
    first, last = logging_dates

    for _, art_id, time_stamp in articles_read:
        # entry[0] = news_paper
        # entry[1] = article_id
        # entry[2] = time_stamp
        if first is None or last is None:
            first = last = time_stamp

        if time_stamp < first:
            first = time_stamp
        if time_stamp > last:
            last = time_stamp

    return (first, last)


def subsample_users(n_users, item_data, data_dir, min_hist_len, remove_unk_arts=True, test_time_thresh=None):

    if isinstance(item_data, dict):
        # use a given item sub sample
        valid_item_ids = set(item_data['all'].keys())
        n_max_items = len(item_data['all'])
    elif isinstance(item_data, int):
        valid_item_ids = set()
        n_max_items = item_data
        item_data = {'all': set()}


    c_articles_raw = Counter() # without threshold
    c_articles_thresh = Counter() # count only those from 'valid' histories

    if test_time_thresh is not None:
        item_data['train'] = set()
        item_data['test'] = set()

    user_dict = OrderedDict()
    unk_arts = defaultdict(int)
    removed_users = 0

    logging_dates = (None, None) # first & last reading date

    for i, user in tqdm(enumerate(data_stream_generator(data_dir + "users"))):
        # quick preliminary eval
        if len(user['articles_read']) >= min_hist_len:
            # remove unknown articles from reading history
            if remove_unk_arts:
                history = []
                # check if article ID is present in our set of valid IDs
                for entry in user['articles_read']:
                    _, art_id, time_stamp = entry

                    if art_id not in valid_item_ids:
                        unk_arts[art_id] += 1

                        # if replace_with_unk:
                        #     entry[1] = unk_art_id
                        #     history.append(entry)
                    else:
                        if test_time_thresh is not None:
                            if 'articles_train' not in user.keys():
                                user['articles_train'] = []
                                user['articles_test'] = []

                            if time_stamp < test_time_thresh:
                                user['articles_train'].append(entry)
                            else:
                                user['articles_test'].append(entry)

                        history.append(entry)
                        c_articles_raw.update([art_id])

                user['articles_read'] = history

            # evaluate length reading history
            if len(user['articles_read']) >= min_hist_len:
                # add valid user that fulfills min read condition
                # item.keys() = dict_keys(['user_id', 'articles_read', 'opened_pushes', 'articles_pushed'])
                user['articles_read'] = sorted(user['articles_read'], key=lambda entry: entry[2])  # sort by time_stamp

                if test_time_thresh is not None:
                    if len(user['articles_train']) >= min_hist_len:
                        #
                        art_train = [art_id for _, art_id, _ in user['articles_train']]
                        item_data['train'].update(art_train)
                        c_articles_thresh.update(art_train)

                        art_test = [art_id for _, art_id, _ in user['articles_test']]
                        item_data['test'].update(art_test)
                        c_articles_thresh.update(art_test)
                    else:
                        continue

                user['n_arts_read'] = len(user['articles_read'])
                user_dict[user['user_id']] = {key: val for (key, val) in user.items() if key != 'user_id'}
                #
                logging_dates = update_logging_dates(user['articles_read'], logging_dates)
            else:
                removed_users += 1
                if removed_users % 1e4 == 0:
                    print("{} \t {}".format(len(user_dict), removed_users))

        # break condition
        if len(user_dict) == n_users or i == n_users * 200: #200
            break
    #print("Users sampled {}".format(len(user_dict)))
    print("\n Start logging date: {} \t End: {}".format(*logging_dates))

    return user_dict, logging_dates


#deprecated [20.04.2020] {HB}
# def get_dpg_data_sample(data_dir, n_news, n_users, snippet_len=30, min_hist_len=5, sample_name=None, save_path=None, test_time_thresh=None):
#     # UNIX time for last entry
#     #datetime.datetime(2019, 12, 31, 23, 59, 59).strftime('%s')
#     # -> '1577833199'
#     # datetime.datetime(2019, 12, 24, 23, 59, 59).strftime('%s')
#     # => '1577228399'
#
#     try:
#         os.listdir(data_dir)
#     except:
#         raise FileNotFoundError("Given data directory is wrong!")
#
#     # subsample items
#     print("Sample items ...")
#     news_data = subsample_items(n_news, snippet_len, data_dir, test_time_thresh=test_time_thresh)
#
#     # subsample users
#     print("Sample users ...")
#     user_data, logging_dates = subsample_users(n_users, news_data, data_dir, min_hist_len, test_time_thresh=test_time_thresh)
#
#     logging_dates = {'start': logging_dates[0], 'end': logging_dates[1]}
#
#     if test_time_thresh is not None:
#         logging_dates['threshold'] = test_time_thresh
#
#     if save_path is not None:
#         save_path = Path(save_path)
#
#         if sample_name is None:
#             sample_name = ('i{}k_u{}k_s{}'.format(int(n_news/1e3), int(n_users/1e3), snippet_len))
#         save_path = save_path / sample_name
#         save_path.mkdir(parents=True, exist_ok=True)
#         fn = 'news_data.pkl'
#         with open(save_path / fn, 'wb') as fout:
#             pickle.dump(news_data, fout)
#
#         with open(save_path / 'user_data.pkl', 'wb') as fout:
#             pickle.dump(user_data, fout)
#
#         with open(save_path / 'logging_dates.json', 'w') as fout:
#             json.dump(logging_dates, fout)
#
#     return news_data, user_data, logging_dates

def count_article_interactions(data_dir, n_users=None):

    c_art_ids = Counter()

    if n_users is not None:
        limit = n_users*10
    else:
        limit = -1

    for i, user in tqdm(enumerate(data_stream_generator(data_dir + "users"))):
        _, art_ids, _ = zip(*user['articles_read'])
        c_art_ids.update(art_ids)

        if i == limit:
            break
        elif i % 1e4 == 0 and i > 0:
            print("{} users evaluated".format(i))

    return c_art_ids


def subsample_items_from_id(data_dir: str, valid_ids: set, news_len: int, n_news: int, add_items=False, keys_to_exclude = ["short_id", "url"], test_time_thresh=None):
    item_dict = OrderedDict()
    item_dict['all'] = {}

    # if test_time_thresh is not None:
    #     item_dict['train'] = {}
    #     item_dict['test'] = {}

    if add_items:
        additional_items = n_news - len(valid_ids)
    else:
        additional_items = 0

    for i, item in enumerate(data_stream_generator(data_dir + "items")):
        # item.keys() = dict_keys(['text', 'pub_date', 'author', 'url', 'short_id'])
        if item['short_id'] in valid_ids or additional_items:

            if news_len is not None:
                item['snippet'] = get_text_snippet(item['text'], news_len)

            item['n_words'] = get_n_words(item['text'])
            val_dict = {key: val for (key, val) in item.items() if key not in keys_to_exclude}
            item_dict['all'][item['short_id']] = val_dict

            #item['pub_date'] = (item['first_pub_date'] if item['first_pub_date'] is not None else item['pub_date'])
            # currently no pub_date available [30.03.]

            if item['short_id'] not in valid_ids:
                additional_items -= 1

        if i % 1e5 == 0 and i>0:
            print("{} items scanned ..".format(i))

    return item_dict

def get_data_common_interactions(data_dir, n_news, n_users, news_len=30, min_hist_len=5, sample_name=None, save_path=None, test_time_thresh=None):
    try:
        os.listdir(data_dir)
    except:
        raise FileNotFoundError("Given data directory is wrong!")

    n_news = int(n_news)
    n_users = int(n_users)

    # subsample items
    print("Determine most common articles based on user interaction ...")

    save_path = Path(save_path)

    if (save_path / "counter_article_ids.pkl").exists():
        with open((save_path / "counter_article_ids.pkl"), 'rb') as fin:
            c_art_ids = pickle.load(fin)
    else:
        c_art_ids = count_article_interactions(data_dir)
        with open(save_path / "counter_article_ids.pkl", 'wb') as fout:
            pickle.dump(c_art_ids, fout)

    valid_article_ids = set([e[0] for e in c_art_ids.most_common(n_news)])

    # get all valid items
    add_items = (True if len(valid_article_ids) < n_news else False) # flag to indicate whether to add random articles. only applies if most common is insufficient
    news_data = subsample_items_from_id(data_dir, valid_article_ids, n_news, news_len, add_items=add_items, test_time_thresh=test_time_thresh)

    # subsample users
    print("Sample users ...")
    user_data, logging_dates = subsample_users(n_users, news_data, data_dir, min_hist_len, test_time_thresh=test_time_thresh)

    logging_dates = {'start': logging_dates[0], 'end': logging_dates[1]}

    if test_time_thresh is not None:
        logging_dates['threshold'] = test_time_thresh

    if save_path is not None:

        if sample_name is None:
            sample_name = ('i{}k_u{}k_s{}'.format(int(n_news/1e3), int(n_users/1e3), news_len))
        save_path = save_path / sample_name
        save_path.mkdir(parents=True, exist_ok=True)

        fn = 'news_data.pkl'
        with open(save_path / fn, 'wb') as fout:
            pickle.dump(news_data, fout)

        with open(save_path / 'user_data.pkl', 'wb') as fout:
            pickle.dump(user_data, fout)

        with open(save_path / 'logging_dates.json', 'w') as fout:
            json.dump(logging_dates, fout)

    return news_data, user_data, logging_dates

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='../datasets/dpg/', help='data path')
    parser.add_argument('--save_path', type=str, default='../datasets/dpg/', help='path to save data')

    parser.add_argument('--size', type=str, default='medium', help='size of dataset')
    parser.add_argument('--n_items', type=int, default=45000, help='number of items')
    parser.add_argument('--n_users', type=int, default=10000, help='number of users')

    #parser.add_argument('--vocab_size', type=int, default=30000, help='vocab')
    parser.add_argument('--time_threshold', type=str, default="24-12-2019-23-59-59", help='date for splitting train/test')

    parser.add_argument('--news_len', type=int, default=30, help='number of words from news body')
    parser.add_argument('--min_hist_len', type=int, default=5, help='minimum number of articles in reading history')

    config = parser.parse_args()


    if config.size in DATA_SIZES.keys():
        n_news, n_users = DATA_SIZES[config.size]
    elif "custom" == config.size:
        n_news = config.n_news
        n_users = config.n_users
    else:
        raise NotImplementedError()

    sample_name = config.size + "_time_split_interactions"

    threshold_date = int(datetime.datetime.strptime(config.time_threshold, '%d-%m-%Y-%H-%M-%S').strftime("%s")) #1577228399

    news_data, user_data, logging_dates = get_data_common_interactions(config.data_dir, n_news, n_users,
                                                                       news_len=config.news_len, min_hist_len=config.min_hist_len,
                                                                       save_path=config.save_path, sample_name=sample_name,
                                                                       test_time_thresh=threshold_date)

    for key, val in logging_dates.items():
        print("{} {}".format(key, datetime.datetime.fromtimestamp(val).strftime('%Y-%m-%d %H:%M:%S')))