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
train_captions, train_capt_labels = split_image_from_caption(ld.load_train_captions())
test_captions, test_capt_labels = split_image_from_caption(ld.load_test_captions())

# %%
from sklearn.feature_extraction.text import TfidfVectorizer

# %%
ti_vectorizer = TfidfVectorizer()
X = ti_vectorizer.fit_transform(train_captions)
train_vectors = X.toarray()
print(ti_vectorizer.get_feature_names())

# %%
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize

# %%
nltk.download('stopwords')
nltk.download('punkt')


# %%
# Making the vocabulary

def extract_text_from_captions(*args):
    """
    Extract the text from arrays of ['image', caption'] elements.
    :param args: Arrays containing image name and caption.
    :return: List of captions.
    """
    text_caps = []
    for a in args:
        for i in a:
            text_caps.append(i[1])
    return text_caps


# %%
stop_words = stopwords.words('spanish')

# %%
set(stop_words)

# %%
captions = extract_text_from_captions(train_captions, test_captions)

# %%
train_data = []

for sent in captions:
    buf = []
    for w in word_tokenize(sent):
        buf.append(w.lower())
    train_data.append(buf)

# %%
from gensim.models import Word2Vec

# %%

model = Word2Vec(train_data, min_count=1, size=100, window=5, sg=1)
