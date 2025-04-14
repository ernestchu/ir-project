import numpy as np
import pandas as pd
import os
import math



from datasets import load_dataset
from datasets import load_from_disk
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
import json
# nltk.download('punkt')
# nltk.download('stopwords')

ds = load_dataset("ernestchu/lrclib-20250319", "default")


lyrics_data = ds["train"]
print(len(lyrics_data))
raw_lyrics = [example["plain_lyrics"] for example in lyrics_data if example["has_plain_lyrics"]]

#load label:
with open("emotion_labels.json", "r") as f:
    label_dict = json.load(f)

labels = []
for i in range(len(raw_lyrics)):
    key = str(i)
    label = label_dict.get(key)
    labels.append(label)


stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")



