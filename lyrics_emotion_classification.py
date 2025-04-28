import os
import pickle
from collections import defaultdict, Counter

import numpy as np
import csv
import joblib
from joblib import load

from tqdm import tqdm
from datasets import load_dataset
import gensim.downloader
from gensim.models import Word2Vec, Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.corpora import Dictionary
from gensim.models.ldamulticore import LdaMulticore

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sentence_transformers import SentenceTransformer
import stanza
from spacy.lang.en.stop_words import STOP_WORDS

#===set seed===
SEED = 42

random.seed(SEED)
np.random.seed(SEED)



# === preprocess setup ===
stanza.download("en")
nlp = stanza.Pipeline("en", processors="tokenize,pos,lemma,ner", use_gpu=True)

def stanza_preprocess(text, lemmatize=True):
    doc = nlp(text)
    tokens = []
    keep_entities = {"PERSON", "ORG", "GPE", "LOC", "EVENT", "FAC", "PRODUCT"}
    entities_to_keep = {ent.text.lower() for ent in doc.ents if ent.type in keep_entities}
    for sent in doc.sentences:
        for word in sent.words:
            token = word.lemma if lemmatize else word.text
            token = token.lower()
            if token in STOP_WORDS:
                continue
            if token in entities_to_keep or word.upos in {"NOUN", "VERB", "ADJ", "ADV", "INTJ", "PROPN"}:
                tokens.append(token)
    return tokens



# === Emotion lexicons and helper funcs===
def save_pickle(path, obj):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

def load_nrc_lexicon(filepath="NRC-Emotion-Lexicon-Wordlevel-v0.92.txt"):
    lexicon = defaultdict(set)
    with open(filepath, "r") as f:
        for line in f:
            word, emotion, assoc = line.strip().split("\t")
            if int(assoc) == 1:
                lexicon[word].add(emotion)
    return lexicon

class NRCEmotionVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lexicon):
        self.lexicon = lexicon
        self.emotions = sorted(set(e for s in lexicon.values() for e in s))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        emotion_vectors = []
        for doc in X:
            counter = Counter()
            for word in doc:
                if word in self.lexicon:
                    for emotion in self.lexicon[word]:
                        counter[emotion] += 1
            v = np.array([counter[e] for e in self.emotions], dtype=float)
            emotion_vectors.append(v)
        return np.vstack(emotion_vectors)


def load_vader_vad_lexicon(filepath="vader_lexicon.txt"):
    vad_lexicon = {}
    with open(filepath, "r") as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 4:
                word = parts[0].lower()
                valence = float(parts[1])  
                vad_lexicon[word] = np.array([valence, valence, valence])
    return vad_lexicon

def load_warriner_vad_lexicon(filepath="warriner_2013_valence.csv"):
    import pandas as pd
    df = pd.read_csv(filepath)
    vad_lexicon = {}
    for _, row in df.iterrows():
        word = row.get('Word')
        if isinstance(word, str):
            word = word.lower()
            vad_lexicon[word] = np.array([
                row.get('V.Mean.Sum', 0.0),
                row.get('A.Mean.Sum', 0.0),
                row.get('D.Mean.Sum', 0.0)
            ])
    return vad_lexicon

class CombinedVADVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, lexicons):
        self.lexicons = lexicons

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        vectors = []
        for doc in X:
            all_scores = []
            for lexicon in self.lexicons:
                scores = [lexicon[word] for word in doc if word in lexicon]
                if scores:
                    v = np.mean(scores, axis=0)
                else:
                    v = np.zeros(3)
                all_scores.append(v)
            combined = np.concatenate(all_scores)
            vectors.append(combined)
        return np.vstack(vectors)


# === general text vectorization classes===

class Word2VecSKL(BaseEstimator, TransformerMixin):
    def __init__(self, model=Word2Vec, pretrained_model="word2vec-google-news-300", model_kwargs={
        'vector_size': 100, 'window': 5, 'min_count': 2, 'workers': 4, 'seed': 42
    }):
        self.model = model
        self.pretrained_model = pretrained_model
        self.model_kwargs = model_kwargs or {}

    def fit(self, X, y=None):
        if self.pretrained_model:
            self.model = gensim.downloader.load(self.pretrained_model)
        else:
            w2v_model = self.model(X, **self.model_kwargs)
            self.model = w2v_model.wv
        self.dim = self.model.vector_size
        return self

    def transform(self, X):
        vectors = []
        for doc in X:
            words = [self.model[w] for w in doc if w in self.model]
            vectors.append(np.mean(words, axis=0) if words else np.zeros(self.dim))
        return np.vstack(vectors)


class Doc2VecSKL(BaseEstimator, TransformerMixin):
    def __init__(self, vector_size=100, window=5, min_count=2, epochs=20):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.epochs = epochs

    def fit(self, X, y=None):
        tagged_data = [TaggedDocument(words=doc, tags=[i]) for i, doc in enumerate(X)]
        self.model = Doc2Vec(vector_size=self.vector_size, window=self.window,
                             min_count=self.min_count, workers=4, epochs=self.epochs, seed=SEED)
        self.model.build_vocab(tagged_data)
        self.model.train(tagged_data, total_examples=self.model.corpus_count, epochs=self.epochs)
        return self

    def transform(self, X):
        return np.vstack([self.model.infer_vector(doc) for doc in X])


class LDAVectorizer(BaseEstimator, TransformerMixin): 
    def __init__(self, num_topics=8, passes=3):
        self.num_topics = num_topics
        self.passes = passes

    def fit(self, X, y=None):
        self.dictionary = Dictionary(X)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in X]
        self.model = LdaMulticore(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=self.passes,
            workers=4,
            random_state=42
        )
        return self

    def transform(self, X):
        corpus = [self.dictionary.doc2bow(doc) for doc in X]
        topic_distributions = []
        for doc_bow in corpus:
            dist = self.model.get_document_topics(doc_bow, minimum_probability=0.0)
            dense = np.array([prob for _, prob in dist])
            topic_distributions.append(dense)
        return np.vstack(topic_distributions)


class SentenceBERTVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, model_name="all-mpnet-base-v2"):
        self.model_name = model_name

    def fit(self, X, y=None):
        self.model = SentenceTransformer(self.model_name)
        return self

    def transform(self, X):
        return self.model.encode(X, show_progress_bar=True)


class DenseTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.toarray() if hasattr(X, "toarray") else X




# === Pipeline  ===
def get_classifier(name):
    if name == "nb": 
        return GaussianNB()
    if name == "rf": 
        return RandomForestClassifier(n_estimators=100, random_state=SEED)
    if name == "xgb": 
        return XGBClassifier(eval_metric='mlogloss', random_state=SEED)
    if name == "svm": 
        return SVC(kernel="rbf", probability=True)
    if name == "svm_linear": 
        return SVC(kernel="linear", probability=True)
    if name == "logreg": 
        return LogisticRegression(max_iter=1000, random_state=SEED)
    if name == "lgbm": 
        return LGBMClassifier(random_state=SEED)
    if name == "voting": return VotingClassifier(
    estimators=[
        ("svm", SVC(kernel="rbf", probability=True)),
        ("logreg", LogisticRegression(max_iter=1000)),
        ("xgb", XGBClassifier(eval_metric='mlogloss')),
    ],
    voting="soft",  
    n_jobs=-1
    )

def get_feature_pipeline(feature, classifier, use_lda=True):
    feats = []
    if feature == "tfidf":
        feats.append(("tfidf", TfidfVectorizer(lowercase=True, stop_words='english', max_df=0.95, min_df=2)))
        use_text = True
    elif feature == "word2vec":
        feats.append(("w2v", Word2VecSKL(model_kwargs={'vector_size': 100, 'window': 5, 'min_count': 2, 'workers': 4, 'seed': SEED})))
        use_text = False
    elif feature == "doc2vec":
        feats.append(("doc2vec", Doc2VecSKL(vector_size=100, window=5, min_count=2, epochs=20)))
        use_text = False
    elif feature == "sbert":
        feats.append(("sbert", SentenceBERTVectorizer()))
        use_text = True
    

    if not feature == "sbert":
        feats.append(("nrc", NRCEmotionVectorizer(nrc_lexicon)))
        feats.append(("vad", combined_vad_vectorizer))

    if use_lda and not use_text:
        feats.append(("lda", LDAVectorizer(num_topics=8, passes=3)))

    pipe = Pipeline([
        ("features", FeatureUnion(feats)),
        ("to_dense", DenseTransformer()) if classifier == "nb" else ("noop", "passthrough"),
        ("clf", get_classifier(classifier))
    ])
    return pipe, use_text

# === Load Dataset & Tokenization Files ===
ds = load_dataset("ernestchu/lyrics-emotion-classification")
train_data, test_data = ds["train"], ds["test"]


if os.path.exists("train_tokens.pkl"):
    train_tokens = load_pickle("train_tokens.pkl")
    test_tokens = load_pickle("test_tokens.pkl")
else:
    train_tokens = [stanza_preprocess(d["lyrics"]) for d in tqdm(train_data, desc="Train")]
    test_tokens = [stanza_preprocess(d["lyrics"]) for d in tqdm(test_data, desc="Test")]
    save_pickle("train_tokens.pkl", train_tokens)
    save_pickle("test_tokens.pkl", test_tokens)

nrc_lexicon = load_nrc_lexicon()
vader_lexicon = load_vader_vad_lexicon("vader_lexicon.txt")
warriner_lexicon = load_warriner_vad_lexicon("warriner_2013_valence.csv")
combined_vad_vectorizer = CombinedVADVectorizer([vader_lexicon, warriner_lexicon])

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform([d["class"] for d in train_data])
y_test = label_encoder.transform([d["class"] for d in test_data])


# === Experiments ===
features = ["tfidf", "word2vec", "doc2vec", "sbert" ]
classifiers = ["nb", "rf", "xgb", "logreg", "lgbm","svm", "svm_linear", "voting"] 

best_score = 0.0
best_pipeline = None
best_config = ("", "")


results = "model_performance_results.csv"
with open(results, mode="w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["Feature", "Classifier", "Accuracy", "Precision", "Recall", "F1"])


# === permutation loop ===
for feat in tqdm(features, desc="Feature Types"):
    for clf in tqdm(classifiers, desc=f"{feat} Classifiers", leave=False):
        print(f"==== {feat.upper()} + {clf.upper()} ====")
        pipeline, use_text = get_feature_pipeline(feat, clf, use_lda=True)
        X_train = [d["lyrics"] for d in train_data] if use_text else train_tokens
        X_test = [d["lyrics"] for d in test_data] if use_text else test_tokens

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)

        print("Accuracy:", accuracy)
        print("Weighted Precision:", precision)
        print("Weighted Recall:", recall)
        print("Weighted F1:", f1)
        print()

        with open(results, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([feat, clf, accuracy, precision, recall, f1])

        if f1 > best_score:
            best_score = f1
            best_pipeline = pipeline
            best_config = (feat, clf)


print(f"Best Model: {best_config[0].upper()} + {best_config[1].upper()} with F1 = {best_score:.4f}")
joblib.dump(best_pipeline, "best_emotion_model.pkl")

loaded_model = load("best_emotion_model.pkl")
