import json
import re

from datasets import load_dataset
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


# nltk.download("punkt_tab")
# nltk.download("stopwords")

# === Load Data ===
ds = load_dataset("ernestchu/lyrics-emotion-classification")
train_data = ds["train"]
dev_data = ds["dev"]
test_data = ds["test"]


pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(
        lowercase=True,
        stop_words='english',
        max_df=0.95,
        min_df=2
    )),
    ('nb', MultinomialNB(alpha=1.0))
])

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

exit()

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


