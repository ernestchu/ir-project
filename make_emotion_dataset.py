import os
import json
import re

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from transformers import pipeline
from datasets import load_dataset, Dataset

LYRICS_JSON = "raw_lyrics.json"
LABELS_JSON = "emotion_labels.json"
DEDUP_SIM_THRES = 0.95
LANG_DET_BATCH_SIZE = 512
SEED = 20250421
DS_REPO = "ernestchu/lyrics-emotion-classification"


def deduplicate(dataset):
    """
    Deduplicate the dataset.
    """
    for d in dataset:
        d['lyrics'] = re.sub(r"\s+", " ", d['lyrics'].replace("\n", " ")).strip().lower()

    # remove exact duplicates
    seen = set()
    uniq_dataset = [d for d in dataset if not (d['lyrics'] in seen or seen.add(d['lyrics']))]

    # remove close duplicates based on cosine similarity
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform([d['lyrics'] for d in uniq_dataset])
    cos_sim_matrix = cosine_similarity(tfidf_matrix)

    seen = set()
    deduplicated_dataset = []
    for i in tqdm(range(len(uniq_dataset)), desc="Deduplicating"):
        if i in seen:
            continue
        deduplicated_dataset.append(uniq_dataset[i])
        for j in range(i + 1, len(uniq_dataset)):
            if cos_sim_matrix[i, j] > DEDUP_SIM_THRES:
                seen.add(j)

    print(f"Keeping {len(deduplicated_dataset)}/{len(dataset)} unique entries.")
    return deduplicated_dataset


def remove_non_english(dataset):
    """
    Remove non-English lyrics from the dataset.
    """
    pipe = pipeline("text-classification", "papluca/xlm-roberta-base-language-detection")

    predicted_langs = pipe([d['lyrics'] for d in dataset], batch_size=LANG_DET_BATCH_SIZE, top_k=1, truncation=True)
    english_dataset = [d for d, lang in zip(dataset, predicted_langs) if lang[0]['label'] == 'en']

    print(f"Keeping {len(english_dataset)}/{len(dataset)} english entries.")
    return english_dataset


def clean_text(dataset):
    """
    Clean the text by removing special characters and extra spaces.
    """
    for d in dataset:
        text = d['lyrics']
        text = re.sub(r"[\n\t]", " ", text)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
        text = re.sub(r"\s+", " ", text)
        d['lyrics'] = text.strip().lower()

    return dataset


def create_emotion_dataset(dataset):
    """
    Get metadata, train/test split and upload the dataset to HF.
    """
    # Get metadata
    track_ids = [d['track_id'] for d in dataset]
    lrclib = load_dataset("ernestchu/lrclib-20250319", split="train", streaming=True)
    num_found_metadata = 0
    pbar = tqdm(total=len(dataset), desc="Getting metadata")
    for track in lrclib:
        if track['id'] in track_ids:
            dataset[track_ids.index(track['id'])].update({
                'track_name': track['name'],
                'artist_name': track['artist_name'],
                'album_name': track['album_name'],
            })
            num_found_metadata += 1
            pbar.update()
        if num_found_metadata == len(dataset):
            break
    pbar.close()

    # make HF dataset
    def generator():
        for d in dataset:
            yield d

    ds = Dataset.from_generator(generator)
    ds = ds.train_test_split(test_size=0.2, seed=SEED)
    final_ds = {'train': ds['train']}
    ds = ds['test'].train_test_split(test_size=0.5, seed=SEED)
    final_ds.update({'dev': ds['train'], 'test': ds['test']})

    for split, ds in final_ds.items():
        ds.push_to_hub(DS_REPO, split=split)

def extract_labels(labels):
    """
    Extract labels from GPT's responses.
    """
    labels = {
        k: v.split(".")[0].strip() for k, v in labels.items() if "." in v
    }
    numeric_labels = {}
    for k in labels:
        try:
            numeric_labels[k] = int(labels[k])
        except ValueError:
            pass
    numeric_labels = {
        k: v for k, v in numeric_labels.items() if v > 0 and v <= 8
    }
    return numeric_labels


if __name__ == "__main__":
    with open(LYRICS_JSON, "r") as f:
        raw_lyrics = json.load(f)

    with open(LABELS_JSON, "r") as f:
        emotion_labels = json.load(f)

    emotion_labels = extract_labels(emotion_labels)
    dataset = [
        {'track_id': int(track_id), 'lyrics': raw_lyrics[track_id], 'class': emotion_labels[track_id]}
        for track_id in raw_lyrics if track_id in emotion_labels]

    print(f"Loaded {len(dataset)}/{len(raw_lyrics)} lyrics with valid labels")

    print("Step 1/4: remove exact duplicates and the lyrics that differs only by punctuation but is essentially the same song using cosine similarity")
    dataset = deduplicate(dataset)

    print("Step 2/4: remove non english songs according to the lyrics")
    dataset = remove_non_english(dataset)

    print("Step 3/4: remove all punctuations for lyrics")
    dataset = clean_text(dataset)

    print("Step 4/4: get song metadata, split the dataset into train/dev/test and upload to HF")
    create_emotion_dataset(dataset)

    print(f"Finished! Processed dataset pushed to https://huggingface.co/datasets/{DS_REPO}")
