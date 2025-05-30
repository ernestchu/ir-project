# ir-project

- [x] Dataset: [lrclib-dump](https://huggingface.co/datasets/ernestchu/lrclib-20250319), [lyrics-emotion-classification](https://huggingface.co/datasets/ernestchu/lyrics-emotion-classification)
- [ ] Preprocess: raw, stopword+stem, LLM (dedup, summarize)
- [ ] Vectorize: tf, word2vec, (or both)
- [ ] Classification: bayesian
- [ ] Assessable task: mood classification, in-lyric search
- [ ] Apps: song recommendation (single track, playlist)



code step: 
1. run [lyrics_emotion_annotation_gpt.py](lyrics_emotion_annotation_gpt.py) to generate emotion cluster label and 
emotion word for the song that has plain lyrics as well as the corresponding raw_lyrics for the song
generate: emotion_labels.json, raw_lyrics.json

2. run [make_lyrics_emotion_dataset.py](make_lyrics_emotion_dataset.py) to perform data preprocessing and upload the [dataset](https://huggingface.co/datasets/ernestchu/lyrics-emotion-classification) to HF. Preprocess steps:
    1. remove exact duplicates and the lyrics that differs only by puntuation but is essentially the same song using cosine similarity
    2. remove non english songs according to the lyrics
    3. remove all punctuations for lyrics
    4. get song metadata, split the dataset into train/dev/test and upload to HF


(37473 annotated samples by GPT, after preprocess we have 20229 unique English songs )

3. run [lyrics_emotion_classification.py](lyrics_emotion_classification.py) to create model for info retreival

Accuracy: 0.5269




Permutation description:


Features Used:
    TF-IDF: Bag-of-words based frequency representation using Scikit-learn’s TfidfVectorizer

    Word2Vec: Trainable embeddings using Gensim's Word2Vec

    Doc2Vec: Paragraph embeddings trained using Gensim's Doc2Vec

    SBERT: Sentence embeddings from all-mpnet-base-v2 (SentenceTransformers)


classifier models:
    Logistic Regression:
    SVM: RBF kernel, Linear kernel
        
    Naive Bayes
    Ranfdom Forest
    Xgboost
    LightGBM
    Voting Stacked model






nlp preprocessing:
stanza from standform 



Emotion Lexicons:
NRC Emotion Lexicon (Mohammad & Turney, 2013)
Provides binary word-to-emotion associations for 8 emotions.

VADER Valence Lexicon
Captures Valence scores used in social media sentiment analysis.

Warriner VAD Lexicon (Warriner et al., 2013)
Provides continuous Valence, Arousal, and Dominance scores for ~14,000 English words.





Limitation:
only retreive and classify english lyrics, can further process non english songs

use gpt to create label, only considering the lyrics, can further expand the project
using wave2vec to cluster songs using the actaul music audio to make the classification more robust.

use other lassifier to model the songs

classification and reccomendation is restricted to the format of the dataset we get, future approach can be done by connecting to api to fetch 
