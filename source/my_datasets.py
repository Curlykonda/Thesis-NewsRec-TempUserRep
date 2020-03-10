from torch.utils.data import Dataset, Dataloader, IterableDataset

class DPG_Dataset(Dataset):

    def __init__(self, data, labels, news_as_word_ids):
        self.data = data # data[user_id] = [user_id, hist, candidates]
        for u_id in data.keys():
            self.labels[u_id] = labels[u_id] # labels[user_id] = [0 0 1 0 0]

        self.news_as_word_ids = news_as_word_ids # mapping from article id to sequence of word ids

    def __getitem__(self, u_id):
        _, hist_article_ids, candidate_ids = self.data[u_id]
        labels = self.labels[u_id]

        hist_as_word_ids = self.news_as_word_ids(hist_article_ids)
        cands_as_word_ids = self.news_as_word_ids(candidate_ids)

        return (hist_as_word_ids, cands_as_word_ids, u_id), labels

    def __len__(self):
        return len(self.labels)

    @property
    def news_size(self):
        return len(self.news_as_word_ids)