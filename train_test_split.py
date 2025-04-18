import json
from sklearn.model_selection import train_test_split

# Load data
with open("raw_lyrics.json", "r") as f:
    raw_lyrics = json.load(f)

with open("emotion_labels.json", "r") as f:
    label_dict = json.load(f)

with open("song_names.json", "r") as f:
    song_name = json.load(f)

# Prepare data lists
lyrics_texts = []
labels = []
songs = []
ids = []

for ID, lyric in raw_lyrics.items():
    label = label_dict.get(ID)
    song = song_name.get(ID)
    if label is not None and lyric is not None and lyric.strip():
        lyrics_texts.append(lyric)
        labels.append(label)
        ids.append(int(ID))
        songs.append(song)

train_lyrics, test_lyrics, train_labels, test_labels, train_songs, test_songs, train_ids, test_ids = train_test_split(
    lyrics_texts, labels, songs, ids, test_size=0.3, random_state=42, stratify=labels
)

# Optionally: save splits to JSON
with open("train.json", "w") as f:
    json.dump({
        "lyrics": train_lyrics,
        "labels": train_labels,
        "songs": train_songs,
        "ids": train_ids
    }, f, indent=2)

with open("test.json", "w") as f:
    json.dump({
        "lyrics": test_lyrics,
        "labels": test_labels,
        "songs": test_songs,
        "ids": test_ids
    }, f, indent=2)

print(f"Train size: {len(train_lyrics)}, Test size: {len(test_lyrics)}")
