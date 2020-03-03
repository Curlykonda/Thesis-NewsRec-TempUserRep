import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import keras
from keras.layers import *
from keras.models import Model
from keras import backend as K
from keras.optimizers import *

from utils_npa import *

from metrics import *

import torch
import torch.nn as nn
import torch.functional as F

from sklearn.metrics import roc_auc_score


npratio = 4
results = []

MAX_SENT_LENGTH = 30
MAX_SENTS = 50

BATCH_SIZE = 100

##user embedding - word & article level
user_id = Input(shape=(1,), dtype='int32')
user_embedding_layer = Embedding(len(userid_dict), 50, trainable=True)
user_embedding = user_embedding_layer(user_id)
user_embedding_word = Dense(200, activation='relu')(user_embedding)
user_embedding_word = Flatten()(user_embedding_word)
user_embedding_news = Dense(200, activation='relu')(user_embedding)
user_embedding_news = Flatten()(user_embedding_news)

##news encoder
news_input = Input(shape=(MAX_SENT_LENGTH,), dtype='int32')
embedding_layer = Embedding(len(word_dict), 300, weights=[embedding_mat], trainable=True)
embedded_sequences = embedding_layer(news_input)
embedded_sequences = Dropout(0.2)(embedded_sequences)

cnnouput = Convolution1D(nb_filter=400, filter_length=3, padding='same', activation='relu', strides=1)(
    embedded_sequences)
cnnouput = Dropout(0.2)(cnnouput)

# personalised attention - word level
attention_a = Dot((2, 1))([cnnouput, Dense(400, activation='tanh')(user_embedding_word)])
attention_weight = Activation('softmax')(attention_a)
news_rep = keras.layers.Dot((1, 1))([cnnouput, attention_weight])
newsEncoder = Model([news_input, user_id], news_rep)

# browsing history as concatenation of MAX_SENTS articles
all_news_input = [keras.Input((MAX_SENT_LENGTH,), dtype='int32') for _ in range(MAX_SENTS)]
browsed_news_rep = [newsEncoder([news, user_id]) for news in all_news_input]
browsed_news_rep = concatenate([Lambda(lambda x: K.expand_dims(x, axis=1))(news) for news in browsed_news_rep], axis=1)

## user encoder
# personalised attention - article level
attention_news = keras.layers.Dot((2, 1))([browsed_news_rep, Dense(400, activation='tanh')(user_embedding_news)])
attention_weight_news = Activation('softmax')(attention_news)
user_rep = keras.layers.Dot((1, 1))([browsed_news_rep, attention_weight_news])

# candidate items - as pseudo K + 1 classification task
candidates = [keras.Input((MAX_SENT_LENGTH,), dtype='int32') for _ in range(1 + npratio)]
candidate_vecs = [newsEncoder([candidate, user_id]) for candidate in candidates]
logits = [keras.layers.dot([user_rep, candidate_vec], axes=-1) for candidate_vec in candidate_vecs]
logits = keras.layers.Activation(keras.activations.softmax)(keras.layers.concatenate(logits))

model = Model(candidates + all_news_input + [user_id], logits)
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['acc'])

candidate_one = keras.Input((MAX_SENT_LENGTH,))
candidate_one_vec = newsEncoder([candidate_one, user_id])
score = keras.layers.Activation(keras.activations.sigmoid)(keras.layers.dot([user_rep, candidate_one_vec], axes=-1))
model_test = keras.Model([candidate_one] + all_news_input + [user_id], score)

def train(model, n_epochs=3, batch_size=100):


    for ep in range(n_epochs):
        traingen = generate_batch_data_train(all_train_pn, all_label, all_train_id, 100) # batch_size
        model.fit_generator(traingen, epochs=1, steps_per_epoch=len(all_train_id) // 100)
        testgen = generate_batch_data_test(all_test_pn, all_test_label, all_test_id, 100)
        click_score = model_test.predict_generator(testgen, steps=len(all_test_id) // 100, verbose=1)

        # test

        all_auc = []
        all_mrr = []
        all_ndcg5 = []
        all_ndcg10 = []

        for m in all_test_index:
            if np.sum(all_test_label[m[0]:m[1]]) != 0 and m[1] < len(click_score):
                all_auc.append(roc_auc_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0]))
                all_mrr.append(mrr_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0]))
                all_ndcg5.append(ndcg_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0], k=5))
                all_ndcg10.append(ndcg_score(all_test_label[m[0]:m[1]], click_score[m[0]:m[1], 0], k=10))
        results.append([np.mean(all_auc), np.mean(all_mrr), np.mean(all_ndcg5), np.mean(all_ndcg10)])
        print(np.mean(all_auc), np.mean(all_mrr), np.mean(all_ndcg5), np.mean(all_ndcg10))


class NPA(torch.nn.Module):

    def __init__(self, dim_uid_emb, dim_pref_query, word_embeddings, dim_word_embeddings, cnn_filter):
        super(NPA, self).__init__()

        raise NotImplementedError()


    def forward(self, *input, **kwargs):
        raise NotImplementedError()