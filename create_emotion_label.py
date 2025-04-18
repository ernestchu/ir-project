import numpy as np
import pandas as pd
import openai
import time
import json
from tqdm import tqdm
from datasets import load_dataset

# === CONFIGURATION ===
CHUNK_START = 27000
CHUNK_END = 30000
LABELS_PATH = "emotion_labels_by_id.json"
RAW_LYRICS_PATH = "raw_lyrics_by_id.json"

client = openai.OpenAI(api_key='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')


try:
    with open(LABELS_PATH, "r") as f:
        emotion_labels = json.load(f)
except FileNotFoundError:
    emotion_labels = {}

try:
    with open(RAW_LYRICS_PATH, "r") as f:
        raw_lyrics = json.load(f)
except FileNotFoundError:
    raw_lyrics = {}


ds = load_dataset("ernestchu/lrclib-20250319", split="train")
ds_chunk = ds.select(range(CHUNK_START, CHUNK_END))
ds_chunk = ds_chunk.filter(lambda x: x["has_plain_lyrics"])


def classify_emotion_gpt(lyrics):
    prompt = (
        "Given the following song lyrics, identify the primary emotion or mood expressed. "
        "Choose only one word from the list below that best matches the mood. Do not invent new words.\n\n"
        "Mood Clusters:\n"
        "1. awe-inspiring, dignified, lofty, sacred, serious, sober, solemn, spiritual\n"
        "2. dark, depressing, doleful, frustrated, gloomy, heavy, melancholy, mournful, pathetic, sad, tragic\n"
        "3. dreamy, longing, plaintive, pleading, sentimental, tender, yearning, yielding\n"
        "4. leisurely, lyrical, quiet, satisfying, serene, soothing, tranquil\n"
        "5. delicate, fanciful, graceful, humorous, light, playful, quaint, sprightly, whimsical\n"
        "6. bright, cheerful, gay, happy, joyous, merry\n"
        "7. agitated, dramatic, exciting, exhilarated, impetuous, passionate, restless, sensational, soaring, triumphant\n"
        "8. emphatic, exalting, majestic, martial, ponderous, robust, vigorous\n\n"
        f"Lyrics:\n{lyrics}\n"
        "Label:"
    )
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print(f"OpenAI error: {e}")
        return None


try:
    for d in tqdm(ds_chunk, desc="Labeling lyrics"):
        track_id = str(d["id"])
        if track_id in emotion_labels:
            continue  
        lyrics = d["plain_lyrics"]
        label = classify_emotion_gpt(lyrics)
        if label:
            emotion_labels[track_id] = label
            raw_lyrics[track_id] = lyrics

        time.sleep(0.7)  

except KeyboardInterrupt:
    print("Interrupted — saving...")

finally:
    with open(LABELS_PATH, "w") as f:
        json.dump(emotion_labels, f, indent=2)
    with open(RAW_LYRICS_PATH, "w") as f:
        json.dump(raw_lyrics, f, indent=2)

    print(f"Done. {len(emotion_labels)} tracks.")
