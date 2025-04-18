import os
import json
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


INPUT_JSON = "raw_lyrics.json"
LABELS_JSON = "emotion_labels.json"
OUTPUT_JSON = "cleaned_lyrics.json"
CLEANED_LABELS_JSON = "cleaned_labels.json"
CHUNK_SIZE = 500
LANGUAGE_LABEL = "en"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


tokenizer = AutoTokenizer.from_pretrained("papluca/xlm-roberta-base-language-detection")
model = AutoModelForSequenceClassification.from_pretrained("papluca/xlm-roberta-base-language-detection")
model.to(DEVICE)
model.eval()


def clean_text(text):
    return re.sub(r"\s+", " ", text.replace("\n", " ")).strip()

def predict_language(texts):
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
    encodings = {k: v.to(DEVICE) for k, v in encodings.items()}
    with torch.no_grad():
        outputs = model(**encodings)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predictions = torch.argmax(probs, dim=1)
        labels = [model.config.id2label[i.item()] for i in predictions]
    return labels


with open(INPUT_JSON, "r") as f:
    raw_lyrics = json.load(f)

try:
    with open(LABELS_JSON, "r") as f:
        emotion_labels = json.load(f)
except FileNotFoundError:
    emotion_labels = {}


seen = set()
unique_lyrics = {k: v for k, v in raw_lyrics.items() if not (v in seen or seen.add(v))}


keys = list(unique_lyrics.keys())
filtered_lyrics = {}
filtered_labels = {}

for i in tqdm(range(0, len(keys), CHUNK_SIZE), desc="Processing chunks"):
    batch_keys = keys[i:i+CHUNK_SIZE]
    batch_texts = [clean_text(unique_lyrics[k]) for k in batch_keys]

    predicted_langs = predict_language(batch_texts)

    for k, text, lang in zip(batch_keys, batch_texts, predicted_langs):
        if lang == LANGUAGE_LABEL:
            filtered_lyrics[k] = text
            if k in emotion_labels:
                filtered_labels[k] = emotion_labels[k]

    
    with open(OUTPUT_JSON, "w") as f:
        json.dump(filtered_lyrics, f, indent=2)
    with open(CLEANED_LABELS_JSON, "w") as f:
        json.dump(filtered_labels, f, indent=2)

print(f"Finished. Saved {len(filtered_lyrics)} English lyrics and {len(filtered_labels)} matching labels.")
