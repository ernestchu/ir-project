import numpy as np
import pandas as pd
import os
import math

import openai
import time
import json
from tqdm import tqdm

from datasets import load_dataset


ds = load_dataset("ernestchu/lrclib-20250319", "default")



lyrics_data = ds["train"]
raw_lyrics = [example["plain_lyrics"] for example in lyrics_data if example["has_plain_lyrics"]]


try:
    with open("emotion_labels.json", "r") as f:
        emotion_labels = json.load(f)
except FileNotFoundError:
    emotion_labels = {}




client = openai.OpenAI(api_key='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') 

def classify_emotion_with_clusters(lyrics_text):
    prompt = (
        "Given the following song lyrics, identify the primary emotion or mood expressed. "
        "Choose only one from the list below that best represents the emotion. "
        "Return only the single word label (e.g. 'cheerful', 'calm', 'frustrated') with one key word from the cluster description that best describe the mood of the lyrics\n\n"
        "Mood Clusters:\n"
        "1. awe-inspiring, dignified, lofty, sacred, serious, sober, solemn, spiritual\n"
        "2. dark, depressing, doleful, frustrated, gloomy, heavy, melancholy, mournful, pathetic, sad, tragic\n"
        "3. dreamy, longing, plaintive, pleading, sentimental, tender, yearning, yielding\n"
        "4. leisurely, lyrical, quiet, satisfying, serene, soothing, tranquil\n"
        "5. delicate, fanciful, graceful, humorous, light, playful, quaint, sprightly, whimsical\n"
        "6. bright, cheerful, gay, happy, joyous, merry\n"
        "7. agitated, dramatic, exciting, exhilarated, impetuous, passionate, restless, sensational, soaring, triumphant\n"
        "8. emphatic, exalting, majestic, martial, ponderous, robust, vigorous\n\n"
        "make sure the words you assign to each id is present in the clusters above do not give words that are not present in the cluster description"
        f"Lyrics:\n{lyrics_text}\n\n"
    )

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=20,
        )
        return response.choices[0].message.content.strip().lower()
    except Exception as e:
        print("Error:", e)
        return None



for i, lyric in tqdm(enumerate(raw_lyrics), total=len(raw_lyrics)):
    if str(i) in emotion_labels:
        continue  

    emotion = classify_emotion_with_clusters(lyric)
    emotion_labels[str(i)] = emotion
    print(f"{i}: {emotion}")

    if i % 10 == 0:
        with open("emotion_labels.json", "w") as f:
            json.dump(emotion_labels, f, indent=2)

    time.sleep(1.1) 


# Final save
with open("emotion_labels.json", "w") as f:
    json.dump(emotion_labels, f, indent=2)

