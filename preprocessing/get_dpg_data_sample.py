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

def data_stream_generator(data_dir):
    files = [file for file in Path(data_dir).iterdir() if
             file.is_file() and file.name[0] not in '_.']
    for file in files:
        with smart_open.open(file) as rf:
            yield from (json.loads(line) for line in rf.read().splitlines() if line)

def get_text_snippet(text, len_snippet):
    naive_tokens = str(text).split(" ")[:len_snippet]
    return " ".join(naive_tokens)

def subsample_items(n_items, snippet_len, data_dir, keys_to_exclude = ["short_id", "url"], test_time_thresh=None):
    item_dict = OrderedDict()
    item_dict['all'] = {}

    # if test_time_thresh is not None:
    #     item_dict['train'] = {}
    #     item_dict['test'] = {}

    for i, item in enumerate(data_stream_generator(data_dir + "items")):
        #item.keys() = dict_keys(['text', 'pub_date', 'author', 'url', 'short_id'])
        if snippet_len is not None:
            item['snippet'] = get_text_snippet(item['text'], snippet_len)

        val_dict = {key: val for (key, val) in item.items() if key not in keys_to_exclude}
        item_dict['all'][item['short_id']] = val_dict

        if item['pub_date']: #< test_time_thresh:
             pass # currently no pub_date available [30.03.]

        if i == n_items-1:
            break

    return item_dict


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


def subsample_users(n_users, item_data, data_dir, min_n_arts, remove_unk_arts=True, test_time_thresh=None):

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
        if len(user['articles_read']) >= min_n_arts:
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
            if len(user['articles_read']) >= min_n_arts:
                # add valid user that fulfills min read condition
                # item.keys() = dict_keys(['user_id', 'articles_read', 'opened_pushes', 'articles_pushed'])
                if test_time_thresh is not None:
                    if len(user['articles_train']) >= min_n_arts:
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


def get_dpg_data_sample(data_dir, n_news, n_users, snippet_len=30, min_hist_len=5, sample_name=None, save_path=None, test_time_thresh=None):
    # UNIX time for last entry
    #datetime.datetime(2019, 12, 31, 23, 59, 59).strftime('%s')
    # -> '1577833199'
    # datetime.datetime(2019, 12, 24, 23, 59, 59).strftime('%s')
    # => '1577228399'

    try:
        os.listdir(data_dir)
    except:
        raise FileNotFoundError("Given data directory is wrong!")

    # subsample items
    print("Sample items ...")
    news_data = subsample_items(n_news, snippet_len, data_dir, test_time_thresh=test_time_thresh)

    # subsample users
    print("Sample users ...")
    user_data, logging_dates = subsample_users(n_users, news_data, data_dir, min_hist_len, test_time_thresh=test_time_thresh)

    logging_dates = {'start': logging_dates[0], 'end': logging_dates[1]}

    if test_time_thresh is not None:
        logging_dates['threshold'] = test_time_thresh

    if save_path is not None:
        save_path = Path(save_path)

        if sample_name is None:
            sample_name = ('i{}k_u{}k_s{}'.format(int(n_news/1e3), int(n_users/1e3), snippet_len))
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
    data_dir = '../datasets/dpg/'
    sample_name = "dev_time_split_test"

    dataset_settings = {'dev': [10e3, 2e3], 'medium': [45e3, 10e3], 'large': [150e3, 50e3]}

    n_news, n_users = dataset_settings['dev']

    threshold_date = (2019, 12, 24, 23, 59, 59)
    #datetime.datetime(2019, 12, 24, 23, 59, 59).strftime('%s')

    news_data, user_data, logging_dates = get_dpg_data_sample(data_dir, n_news, n_users, save_path=data_dir,
                                                              sample_name=sample_name, test_time_thresh=1577228399)
    for key, val in logging_dates.items():
        print("{} {}".format(key, datetime.datetime.utcfromtimestamp(val).strftime('%Y-%m-%d %H:%M:%S')))