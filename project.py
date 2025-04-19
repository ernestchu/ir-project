import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn import cosine_similarity
import nltk

nltk.download("punkt_tab")
nltk.download("stopwords")

# === Load Data ===
with open("train_split.json", "r") as f:
    train_data = json.load(f)

with open("test_split.json", "r") as f:
    test_data = json.load(f)

# tokenize 
stop_words = set(stopwords.words("english"))

def tokenize(text):
    tokens = word_tokenize(text)
    return tokens

def remove_stopwords(tokens):
    tokens = [w for w in tokens if w not in stop_words]

# For word2vec 
def preprocess_texts(texts):
    return [" ".join(tokenize(t)) for t in texts]




# TF-IDF 
def tfidf_vectors(train_texts):
    tfidf = TfidfVectorizer()
    tfidf_train = tfidf.fit_transform(train_texts)
    return tfidf_train

# Word2Vec 
#not done with this part yet need to think about how to further remove unmeaning full words before calling word2vec function 

def word2vec(lyrics):
    model = Word2Vec(lyrics)
    return model



#Bayesian Classifier







#experiment pipeline

tfidf_train = tfidf_vectors()
word2vec_model = word2vec()

#get tfidf for without stop word



