import re
import requests
import concurrent.futures
import pylcs
import random

from tqdm.auto import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from huggingface_hub import snapshot_download
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LsiModel
from gensim.similarities import Similarity

# --- parameters ---
USE_LSI = False
USE_LCS = True

USER = 'ernestchu'
REPO = "lrclib_gensim_assets_lsi_backup" if USE_LSI else "lrclib_gensim_assets"
LRCLIB_API_URL = 'https://lrclib.net/api'
QUERY_TOKEN_LENGTH = 30
TOP_K_SIMILARITY = 1024
NUM_FETCH_WORKERS = 128

# --- download the assets ---
snapshot_download(repo_id=f"{USER}/{REPO}", local_dir=REPO)

bert_tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-uncased")

### Search

def setup():
    # -- load the dictionary and TF–IDF model
    dictionary   = Dictionary.load(f"{REPO}/wordids.dict")
    tfidf_model  = TfidfModel.load(f"{REPO}/tfidf_model.bin")
    if USE_LSI:
        lsi_model  = LsiModel.load(f"{REPO}/lsi_model.bin")
    with open(f"{REPO}/lrclib_doc_ids.txt", "r") as f:
        doc_ids = [int(i) for i in f.read().split('\n')]

    # -- load similarity index
    index = Similarity.load(f"{REPO}/similarity.index")

    def search_query(query):
        query_tokens = bert_tokenizer.tokenize(query)
        query_bow = dictionary.doc2bow(query_tokens)

        query_vec  = tfidf_model[query_bow]
        if USE_LSI:
            query_vec  = lsi_model[query_vec]

        # -- compute cosine similarities
        sims = index[query_vec]  # numpy array of floats
        sims = sorted(enumerate(sims), key=lambda item: -item[1])  # descending
        # get first k results
        sims = sims[:TOP_K_SIMILARITY]
        # convert index to doc_id
        sims = [[doc_ids[doc_pos], float(score)] for doc_pos, score in sims]
        if not USE_LCS:
            return sims

        # Prepare URLs
        urls = [f'{LRCLIB_API_URL}/get/{sim[0]}' for sim in sims]

        # Function to fetch and clean a single lyric
        def fetch_lyric(url):
            try:
                track = requests.get(url).json()
                if 'statusCode' in track and track['statusCode'] == 404:
                    return ''
                lyric = ''
                for key in ['plainLyrics', 'name', 'artistName', 'albumName']:
                    if key in track and isinstance(track[key], str):
                        lyric += track[key]
                return re.sub(r"\s+", "", lyric.lower())
            except Exception as e:
                print(f"Error fetching {url}: {e}")
                return ''

        # Parallel fetching
        with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_FETCH_WORKERS) as executor:
            lyrics = list(executor.map(fetch_lyric, urls))

        for sim, lyric in zip(sims, lyrics):
            sim.append(pylcs.lcs(re.sub(r"\s+", "", query.lower()), lyric))

        sims = sorted(sims, key=lambda item: -item[2])  # descending

        return sims

    return search_query

search_query = setup()


pred = {
    1: [],
    5: [],
    10: [],
    20: [],
    50: [],
}
for row in tqdm(load_dataset("ernestchu/lrclib-20250319", 'random-1k', split='train')):
    id = row['id']
    lyric = row['plain_lyrics']
    tokens = bert_tokenizer.tokenize(lyric)
    start = random.randint(0, len(tokens) - QUERY_TOKEN_LENGTH)
    tokens = tokens[start:start + QUERY_TOKEN_LENGTH]
    token_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
    decoded_text = bert_tokenizer.decode(token_ids)
    results = search_query(decoded_text)
    results = [r[0] for r in results]

    for top_k in pred:
        if id in results[:top_k]:
            pred[top_k].append(1)
        else:
            pred[top_k].append(0)


print(f"USE_LSI: {USE_LSI}")
print(f"USE_LCS: {USE_LCS}")
for top_k in pred:
    acc = sum(pred[top_k]) / len(pred[top_k])
    print(f"Top-{top_k}: {acc}")

