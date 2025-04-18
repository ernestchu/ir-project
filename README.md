# ir-project

- [x] Dataset: [lrclib-dump](https://huggingface.co/datasets/ernestchu/lrclib-20250319)
- [ ] Preprocess: raw, stopword+stem, LLM (dedup, summarize)
- [ ] Vectorize: tf, word2vec, (or both)
- [ ] Classification: bayesian
- [ ] Assessable task: mood classification, in-lyric search
- [ ] Apps: song recommendation (single track, playlist)


code step: 
1. rum create_emotion_label.py to generate emotion cluster label and 
emotion word for the song that has plain lyrics as well as the corresponding raw_lyrics for the song
generate: emotion_labels.json, raw_lyrics.json



2. run preprocess.py to remove non english songs according to the lyrics as well as exact duplicates
generate: cleaned_label.json, cleaned_lyrics.json  

3. run preprocess2.py to remove the lyrics that differs only by punctuation but is essentially the same 
    song using cosine similarity
generate: cleaned_label_cosine.json, cleaned_lyrics cosine.json  

4. run process3.py to remove all punctuations for lyrics, extract the label for testing, and get corresponding song names for the ids
generate: cleaned_lyrics_strict.json, cluster_labels.json (pure song id: label, no emotion word), song_name.json


5. run train_test_split.py to get training and testing data
generate:

run proj.py to create model for info retreival 



(use 32000 samples from data, after preprocess we have 17985 unique songs )


limitation:
only retreive and classify english lyrics, can further process non english songs

use gpt to create label, only considering the lyrics, can further expand the project
using wave2vec to cluster songs using the actaul music audio to make the classification more robust.

use other lassifier to model the songs
