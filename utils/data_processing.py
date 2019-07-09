import numpy as np
import utils.load_data as ld


# %%
# Vectorization of descriptions
def split_image_from_caption(captions_arr):
    """
    # Split and [img, caption] array into two different arrays.
    :param captions_arr: Arrays containing captions and images.
    :return: Captions array and images array in a tuple.
    """
    img = []
    cap = []
    for x, y in captions_arr:
        cap.append(y)
        img.append(x)
    return cap, img


# %%
# Making Dataset for Regression

def make_targets(captions, img_vectors):
    targets = []
    for i in range(len(captions)):
        index = i // 5
        targets.append(img_vectors[index])

    targets = np.array(targets)
    return targets


# %%
train_captions, train_capt_labels = split_image_from_caption(ld.load_train_captions())
test_A_captions, test_A_capt_labels = split_image_from_caption(ld.load_test_A_captions())
test_B_captions, test_B_capt_labels = split_image_from_caption(ld.load_test_B_captions())
test_C_captions, test_C_capt_labels = split_image_from_caption(ld.load_test_C_captions())

train_img_name_vectors, train_img_vectors = ld.load_train_vectors()
test_A_img_name_vectors, test_A_img_vectors = ld.load_test_A_vectors()
test_B_img_name_vectors, test_B_img_vectors = ld.load_test_B_vectors()
test_C_img_name_vectors, test_C_img_vectors = ld.load_test_C_vectors()

# %%
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize, RegexpTokenizer


# nltk.download('stopwords')
# nltk.download('punkt')

# %%
def tokenize(caption_list, max_freq=0, min_freq=0, lower=True):
    res = []
    all_words = []
    dropped = []
    rg_tokenizer = RegexpTokenizer(r'\w+')
    n_captions = 0
    for caption in caption_list:
        cap = []
        tk_cap = rg_tokenizer.tokenize(caption)
        n_captions += 1
        for w in tk_cap:
            _w = w.lower() if lower else w
            cap.append(_w)
            all_words.append(_w)
        res.append(cap)
    all_words = nltk.FreqDist(all_words)
    if max_freq == min_freq == 0:
        return res, all_words, dropped
    else:
        for i in range(n_captions):
            n_w = len(res[i]) - 1
            while n_w >= 0:
                if not (min_freq <= all_words[res[i][n_w]] <= max_freq):
                    dropped.append(res[i].pop(n_w))
                n_w -= 1
        return res, all_words, dropped


# %%
all_captions, _, d = tokenize(train_captions, max_freq=14000, min_freq=2)

# %%
all_words = [word for capt in all_captions for word in capt]
all_words = nltk.FreqDist(all_words)

least_freq = list(filter(lambda x: x[1] == 1, all_words.items()))


# %%

def captions2vec(captions, word_vectors):
    captions_vec = []
    for caption in captions:
        cap = []
        for word in caption:
            if word in word_vectors:
                cap.append(word_vectors[word])
        cap_vec = np.sum(np.array(cap), axis=0)
        cap_vec = cap_vec / np.sqrt(cap_vec.dot(cap_vec))
        captions_vec.append(cap_vec)
    return np.array(captions_vec)
