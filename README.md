# ir-project

- [x] Dataset: [lrclib-dump](https://huggingface.co/datasets/ernestchu/lrclib-20250319), [lyrics-emotion-classification](https://huggingface.co/datasets/ernestchu/lyrics-emotion-classification)
- [ ] Preprocess: raw, stopword+stem, LLM (dedup, summarize)
- [ ] Vectorize: tf, word2vec, (or both)
- [ ] Classification: bayesian
- [ ] Assessable task: mood classification, in-lyric search
- [ ] Apps: song recommendation (single track, playlist)


code step: 
1. rum create_emotion_label.py to generate emotion cluster label and 
emotion word for the song that has plain lyrics as well as the corresponding raw_lyrics for the song
generate: emotion_labels.json, raw_lyrics.json

2. run make_emotion_dataset.py to perform data preprocessing and upload the [dataset](https://huggingface.co/datasets/ernestchu/lyrics-emotion-classification) to HF. Preprocess steps:
    1. remove exact duplicates and the lyrics that differs only by puntuation but is essentially the same song using cosine similarity
    2. remove non english songs according to the lyrics
    3. remove all punctuations for lyrics
    4. get song metadata, split the dataset into train/dev/test and upload to HF


run project.py to create model for info retreival 

(use 37473 samples from data, after preprocess we have 20229 unique English songs )


limitation:
only retreive and classify english lyrics, can further process non english songs

use gpt to create label, only considering the lyrics, can further expand the project
using wave2vec to cluster songs using the actaul music audio to make the classification more robust.

use other lassifier to model the songs
