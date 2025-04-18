import os
import json
import re
from tqdm import tqdm

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


INPUT_LYRICS = "cleaned_lyrics.json"
INPUT_LABELS = "cleaned_labels.json"
OUTPUT_LYRICS = "cleaned_lyrics_cosine.json"
OUTPUT_LABELS = "cleaned_labels_cosine.json"
SIMILARITY_THRESHOLD = 0.95


def clean_text(text):
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip().lower()


with open(INPUT_LYRICS, "r") as f:
    raw_lyrics = json.load(f)

lyrics_keys = list(raw_lyrics.keys())
lyrics_texts = [clean_text(raw_lyrics[k]) for k in lyrics_keys]

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(lyrics_texts)
cos_sim_matrix = cosine_similarity(tfidf_matrix)


seen = set()
deduplicated_lyrics = {}

for i in tqdm(range(len(lyrics_texts)), desc="Deduplicating"):
    if i in seen:
        continue
    deduplicated_lyrics[lyrics_keys[i]] = raw_lyrics[lyrics_keys[i]]
    for j in range(i + 1, len(lyrics_texts)):
        if cos_sim_matrix[i, j] > SIMILARITY_THRESHOLD:
            seen.add(j)

print(f"Deduplicated from {len(lyrics_texts)} to {len(deduplicated_lyrics)} unique entries.")



with open(INPUT_LABELS, "r") as f:
    all_labels = json.load(f)
filtered_labels = {
    k: all_labels[k] for k in deduplicated_lyrics if k in all_labels
}

with open(OUTPUT_LYRICS, "w") as f:
    json.dump(deduplicated_lyrics, f, indent=2)

with open(OUTPUT_LABELS, "w") as f:
    json.dump(filtered_labels, f, indent=2)

print(f"Done! Saved {len(filtered_labels)} cleaned lyrics and labels.")
