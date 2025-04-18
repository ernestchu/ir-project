import json
import re
from tqdm import tqdm
from datasets import load_dataset

INPUT_LYRICS = "cleaned_lyrics_cosine.json"
OUTPUT_LYRICS = "cleaned_lyrics_strict.json"  
INPUT_LABELS = "cleaned_labels_cosine.json"
OUTPUT_LABELS = "cluster_labels.json"



#=================== Remove punctuation ====================#
def remove_all_punctuation(text):
   
    text = re.sub(r"[\n\t]", " ", text)
    
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


with open(INPUT_LYRICS, "r") as f:
    lyrics_data = json.load(f)

print(f"Loaded {len(lyrics_data)} lyrics.")


strict_cleaned_lyrics = {}
for k, v in tqdm(lyrics_data.items(), desc="Stripping punctuation"):
    strict_cleaned_lyrics[k] = remove_all_punctuation(v)


with open(OUTPUT_LYRICS, "w") as f:
    json.dump(strict_cleaned_lyrics, f, indent=2)

print(f"Saved {len(strict_cleaned_lyrics)} stripped lyrics to {OUTPUT_LYRICS}")


#======================= Extract label ============================#

with open(INPUT_LABELS, "r") as f:
    full_labels = json.load(f)

cluster_labels = {
    k: label.split(".")[0].strip() for k, label in full_labels.items() if "." in label
}

with open(OUTPUT_LABELS, "w") as f:
    json.dump(cluster_labels, f, indent=2)

print(f"Extracted {len(cluster_labels)} numeric cluster labels and saved to {OUTPUT_LABELS}")


#===================== Extract song names =========================#

import json
from tqdm import tqdm
from datasets import load_dataset

INPUT_LABELS = "cluster_labels.json"
OUTPUT_NAMES = "song_names.json"

# Load only the song IDs from labels
with open(INPUT_LABELS, "r") as f:
    emotion_labels = json.load(f)
label_ids = set(emotion_labels.keys())


ds_iter = load_dataset("ernestchu/lrclib-20250319", split="train", streaming=True)

song_names_by_id = {}

for row in tqdm(ds_iter, desc="Extracting song names"):
    track_id = str(row["id"])
    if track_id in label_ids:
        song_names_by_id[track_id] = row["name"]


for track_id in label_ids:
    song_names_by_id.setdefault(track_id, "UNKNOWN")

with open(OUTPUT_NAMES, "w") as f:
    json.dump(song_names_by_id, f, indent=2)

print(f"Saved {len(song_names_by_id)} song names to {OUTPUT_NAMES}")

