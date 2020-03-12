import numpy as np

from torch.distributions import transforms
from torch.utils.data import Dataset

class DPG_Dataset(Dataset):

    def __init__(self, data, labels, news_as_word_ids):
        #self.data = data # data[user_id] = [user_id, hist, candidates]
        #self.labels = {} # store labels only relevant for the data subset

        u_id, hist_art_ids, candidate_ids, lbls = zip(*[data_p.values() for data_p in data]) # Note that data format can be converted to Tensors

        self.labels = {u_id : lbl for u_id, lbl in list(zip(u_id, lbls))}

        self.data = list(zip(u_id, hist_art_ids, candidate_ids))

        self.news_as_word_ids = news_as_word_ids # mapping from article id to sequence of word ids

        #self.transform = transforms.Compose([transforms.ToTensor()])

    def __getitem__(self, idx):
        u_id, hist_article_ids, candidate_ids = self.data[idx]
        labels = self.labels[u_id]

        hist_as_word_ids = self.news_as_word_ids[hist_article_ids]
        cands_as_word_ids = self.news_as_word_ids[candidate_ids]
        #u_id = np.expand_dims(u_id, axis=1)

        sample = {'input': (hist_as_word_ids, cands_as_word_ids, u_id), 'labels': np.array(labels)}

        return sample

    def __len__(self):
        return len(self.data)

    @property
    def news_size(self):
        return len(self.news_as_word_ids)