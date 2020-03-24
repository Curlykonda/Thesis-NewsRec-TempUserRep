import json
import gzip
import os
import pickle
from pathlib import Path
import smart_open
import random

from collections import OrderedDict, defaultdict
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

def subsample_items(n_items, snippet_len, data_dir, keys_to_exclude = ["short_id", "url"]):
    item_dict = OrderedDict()

    for i, item in enumerate(data_stream_generator(data_dir + "items")):
        # data[i] = item
        item_dict[item['short_id']] = {key: val for (key, val) in item.items() if key not in keys_to_exclude}

        if snippet_len != None:
            # print(item['text'])
            item_dict[item['short_id']]['snippet'] = get_text_snippet(item['text'], snippet_len)

        if i == n_items - 1:
            break

    return item_dict


def subsample_users(n_users, item_data, data_dir, min_n_arts, remove_unk_arts=True):
    valid_item_ids = set(item_data.keys())
    user_data = []

    user_dict = OrderedDict()
    unk_arts = defaultdict(int)
    removed_users = 0

    for i, item in tqdm(enumerate(data_stream_generator(data_dir + "users"))):
        # quick preliminary eval
        if len(item['articles_read']) >= min_n_arts:
            # remove unknown articles from reading history
            if remove_unk_arts:
                history = []
                # check if article ID is present in our set of valid IDs
                for entry in item['articles_read']:
                    art_id = entry[1]

                    if art_id not in valid_item_ids:
                        unk_arts[art_id] += 1

                        # if replace_with_unk:
                        #     entry[1] = unk_art_id
                        #     history.append(entry)
                    else:
                        history.append(entry)
                item['articles_read'] = history

            # evaluate length reading history
            if len(item['articles_read']) < min_n_arts or len(item['articles_read']) == 0:
                removed_users += 1
                # if removed_users % 1e4 == 0:
                #     print(removed_users)
            else:
                # add valid user that fulfills min read condition
                user_dict[item['user_id']] = {key: val for (key, val) in item.items() if key != 'user_id'}
                user_dict[item['user_id']]['n_articles_read'] = len(user_dict[item['user_id']]['articles_read'])

        # break condition
        if len(user_dict) == n_users or i == n_users * 100:
            break
    #print("Users sampled {}".format(len(user_dict)))

    return user_dict


def get_dpg_data_sample(data_dir, n_news, n_users, snippet_len=30, min_hist_len=5, sample_name=None, save_path=None):

    try:
        os.listdir(data_dir)
    except:
        raise FileNotFoundError("Given data directory is wrong!")

    # subsample items
    news_data = subsample_items(n_news, snippet_len, data_dir)

    # subsample users
    user_data = subsample_users(n_users, news_data, data_dir, min_hist_len)

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

    return news_data, user_data

if __name__ == "__main__":
    data_dir = '../datasets/dpg/'
    n_news = int(100e3)
    n_users = int(50e3)

    news_dat, user_dat = get_dpg_data_sample(data_dir, n_news, n_users, save_path=data_dir)