import numpy as np

# %%

from utils.data_processing import (
    train_captions, train_capt_labels,
    test_A_captions, test_A_capt_labels,
    test_B_captions, test_B_capt_labels,
    test_C_captions, test_C_capt_labels,
    train_img_name_vectors, train_img_vectors,
    test_A_img_name_vectors, test_A_img_vectors,
    test_B_img_name_vectors, test_B_img_vectors,
    test_C_img_name_vectors, test_C_img_vectors
)

# %%
# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
# ti_vectorizer = TfidfVectorizer(max_df=0.7, min_df=0.0001, ngram_range=(1, 2))
# ti_vectorizer = TfidfVectorizer(max_df=0.7, min_df=0.00002)
ti_vectorizer = TfidfVectorizer(max_df=14000, min_df=2)
ti_vectorizer.fit(train_captions)
X = ti_vectorizer.transform(train_captions)
x_test_A = ti_vectorizer.transform(test_A_captions)
x_test_B = ti_vectorizer.transform(test_B_captions)
x_test_C = ti_vectorizer.transform(test_C_captions)
words = ti_vectorizer.get_feature_names()

# %%
# SkipGram
from utils.data_processing import tokenize
import gensim

# %%
import logging

# allows display info
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# %%
sg_vec_size = 500
path_to_keyed_vectors_file = 'vectors/sg.kv'
tok_train_captions = tokenize(train_captions)
tok_test_A_captions = tokenize(test_A_captions)
tok_test_B_captions = tokenize(test_B_captions)
tok_test_C_captions = tokenize(test_C_captions)

# %%
sg_model = gensim.models.Word2Vec(sentences=tok_train_captions, sg=1, size=sg_vec_size, window=5, min_count=2,
                                  sample=1e-5)

# %%
# Save representations
sg_model.wv.save(path_to_keyed_vectors_file)

# %%
# Load representations
words_embeddings = gensim.models.KeyedVectors.load(path_to_keyed_vectors_file, mmap="r")

# %%
from utils.data_processing import captions2vec

x_train = captions2vec(tok_train_captions, words_embeddings)

# %%
x_test_A = captions2vec(tok_test_A_captions, words_embeddings)

# %%
from sklearn.preprocessing import scale

x_train = scale(x_train)
x_test_A = scale(x_test_A)

# %%
# Making Dataset for Regression

x_train = X.toarray()
x_test_A = x_test_A.toarray()
x_test_B = x_test_B.toarray()
x_test_C = x_test_C.toarray()

# %%
# Make Targets
from utils.data_processing import make_targets

y_train = make_targets(train_captions, train_img_vectors)
y_test_A = make_targets(test_A_captions, test_A_img_vectors)
y_test_B = make_targets(test_B_captions, test_B_img_vectors)
y_test_C = make_targets(test_C_captions, test_C_img_vectors)

# %%
# Building Regression
from keras.models import load_model
from keras import Sequential
from keras.layers import Dense, InputLayer, CuDNNLSTM
from keras.layers import Activation, Dropout

# %%
mlp = Sequential()
mlp.add(InputLayer(input_shape=x_train.shape[1:]))

# mlp.add(Dense(units=4096))
mlp.add(Dense(units=512))
mlp.add(Activation(activation='relu'))
mlp.add(Dropout(rate=0.2))

mlp.add(Dense(units=2048))

# %%
mlp.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

history = mlp.fit(x=x_train, y=y_train, batch_size=512, epochs=10, validation_split=0.2)

# %%
mlp.evaluate(x_test_A, y_test_A)

# %%
mlp.evaluate(x_test_B, y_test_B)

# %%
mlp.evaluate(x_test_C, y_test_C)

# %%
y_pred_A = mlp.predict(x_test_A)
y_pred_B = mlp.predict(x_test_B)
y_pred_C = mlp.predict(x_test_C)

# %%
import os
import pickle

vec_path = 'vectors'
pred_A_path = 'y_pred_tf_idf_test_A.pickle'
pred_B_path = 'y_pred_tf_idf_test_B.pickle'
pred_C_path = 'y_pred_tf_idf_test_C.pickle'

# %%
# Saving
with open(os.path.join(vec_path, pred_A_path), 'wb') as o_file:
    pickle.dump(y_pred_A, o_file)

with open(os.path.join(vec_path, pred_B_path), 'wb') as o_file:
    pickle.dump(y_pred_B, o_file)

with open(os.path.join(vec_path, pred_C_path), 'wb') as o_file:
    pickle.dump(y_pred_C, o_file)

# %%
# Loading
with open(os.path.join(vec_path, pred_A_path), 'rb') as o_file:
    y_pred_A = pickle.load(o_file)

with open(os.path.join(vec_path, pred_B_path), 'rb') as o_file:
    y_pred_B = pickle.load(o_file)

with open(os.path.join(vec_path, pred_C_path), 'rb') as o_file:
    y_pred_C = pickle.load(o_file)


# %%
# Computing distances
def get_metrics(pred_list, img_name_list, img_caption_list, targets, img_list):
    def metrics_for_one(pred, img_name, img_caption, targets, img_list):
        met = {
            'img_name': img_name,
            'img_caption': img_caption,
            'pred_vec': pred
        }
        n_img = len(img_list)
        dist = []
        dist_buf = np.sqrt(np.sum(np.square(targets - pred), axis=1))
        for i in range(n_img):
            dist.append([dist_buf[i], img_list[i]])
        dist.sort(key=lambda x: x[0])
        met['nearest'] = dist[0]
        for i, e in enumerate(dist):
            if e[1] == img_name:
                met['position'] = i
                met['distance'] = e[0]
                break
        return met

    n_pred = len(pred_list)
    pred_metrics = []

    for i in range(n_pred):
        m_buffer = metrics_for_one(pred_list[i], img_name_list[i], img_caption_list[i], targets, img_list)
        pred_metrics.append(m_buffer)

    histogram = np.zeros(1000)
    mrr = 0
    mean_pos = 0
    re_call_1 = 0
    re_call_5 = 0
    re_call_10 = 0
    for pm in pred_metrics:
        pos = pm['position']
        mean_pos += (pos + 1)
        mrr += (1 / (pos + 1))
        histogram[pos] += 1
        # Recall at 1
        if pos == 0:
            re_call_1 += 1
        # Recall at 5
        if pos < 5:
            re_call_5 += 1
        # Recall at 10
        if pos < 10:
            re_call_10 += 1
    mrr /= 5000
    mean_pos /= 5000
    re_call_1 /= 5000
    re_call_5 /= 5000
    re_call_10 /= 5000
    metrics = {
        'histogram': histogram,
        'mean_pos': mean_pos,
        're_call_1': re_call_1,
        're_call_5': re_call_5,
        're_call_10': re_call_10,
        'mrr': mrr
    }
    return metrics


# %%
metrics_A = get_metrics(y_pred_A, test_A_capt_labels, test_A_captions, test_A_img_vectors, test_A_img_name_vectors)
metrics_B = get_metrics(y_pred_B, test_B_capt_labels, test_B_captions, test_B_img_vectors, test_B_img_name_vectors)
metrics_C = get_metrics(y_pred_C, test_C_capt_labels, test_C_captions, test_C_img_vectors, test_C_img_name_vectors)


# %%
def print_metrics(metrics):
    for k, v in metrics.items():
        print('{}: {}'.format(k, v))


# %%
print_metrics(metrics=metrics_A)

# %%
# %matplotlib inline
import matplotlib.pyplot as plt

# %%
plt.bar(range(1, 101), metrics_A['histogram'][:100], width=0.5)
plt.show()

# %%
print("Valores en History: {}".format(history.history.keys()))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('categorical_accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()
