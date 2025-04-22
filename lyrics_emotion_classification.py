import json
import re

import numpy as np
from datasets import load_dataset
import gensim.downloader
from gensim.models import Word2Vec
# from nltk.tokenize import word_tokenize
# from nltk.corpus import stopwords
# import nltk

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


# nltk.download("punkt_tab")
# nltk.download("stopwords")

def get_tfidf_mnb_pipeline():
    vec = TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        max_df=0.95,
        min_df=2,
    )
    return Pipeline([
        ('vec', vec),
        ('nb', MultinomialNB(alpha=1.0))
    ])

class Word2VecSKL(BaseEstimator, TransformerMixin):
    def __init__(self, model, pretrained_model, model_kwargs):
        self.model = model
        self.pretrained_model = pretrained_model
        self.model_kwargs = model_kwargs

    def fit(self, X, y=None):
        if self.pretrained_model is not None:
            # Load the pretrained model
            self.model = gensim.downloader.load(self.pretrained_model)
            self.dim = self.model.vector_size
            return self
        X = [x.lower().split() for x in X]
        self.model = self.model(X, **self.model_kwargs)
        self.dim = self.model.vector_size
        return self

    def transform(self, X):
        vecs = []
        for doc in X:
            words = doc.lower().split()
            # collect vectors for words in our vocab
            if isinstance(self.model, Word2Vec):
                good_words = [self.model.wv[w] for w in words if w in self.model.wv]
            else:
                good_words = [self.model[w] for w in words if w in self.model]
            if good_words:
                vecs.append(np.mean(good_words, axis=0))
            else:
                # if no words are in the vocab, use a zero vector
                vecs.append(np.zeros(self.dim, dtype=float))
        return np.vstack(vecs)

def get_word2vec_gnb_pipeline(pretrained_model=None):
    vec = Word2VecSKL(Word2Vec, pretrained_model, {
        'vector_size': 100,   # dimensionality of the embeddings
        'window': 5,
        'min_count': 2,       # ignore very rare words
        'workers': 4,
        'seed': 42
    })
    return Pipeline([
        ('vec', vec),
        ('nb', GaussianNB())
    ])

# === Load Data ===
ds = load_dataset("ernestchu/lyrics-emotion-classification")
train_data = ds["train"]
dev_data = ds["dev"]
test_data = ds["test"]

pipelines = {
    "tfidf-mnb": get_tfidf_mnb_pipeline(),
    "word2vec-gnb": get_word2vec_gnb_pipeline(),
    "word2vec-gnb-google-news-300": get_word2vec_gnb_pipeline('word2vec-google-news-300'),
}

for pipe in  pipelines:
    print(f'======== {pipe} ========')
    pipeline = pipelines[pipe]

    # 3. Train
    pipeline.fit(
        [d['lyrics'] for d in train_data],
        [d['class'] for d in train_data],
    )

    # 4. Predict on test set
    y_pred = pipeline.predict([d['lyrics'] for d in test_data])

    # 5. Evaluate
    y_test = [d['class'] for d in test_data]
    print("Accuracy:", accuracy_score(y_test, y_pred))
    # print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print()

