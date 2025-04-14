import json
import re

# Define your mood clusters and keywords
mood_clusters = {
    1: ["awe-inspiring", "dignified", "lofty", "sacred", "serious", "sober", "solemn", "spiritual"],
    2: ["dark", "depressing", "doleful", "frustrated", "gloomy", "heavy", "melancholy", "mournful", "pathetic", "sad", "tragic"],
    3: ["dreamy", "longing", "plaintive", "pleading", "sentimental", "tender", "yearning", "yielding"],
    4: ["leisurely", "lyrical", "quiet", "satisfying", "serene", "soothing", "tranquil"],
    5: ["delicate", "fanciful", "graceful", "humorous", "light", "playful", "quaint", "sprightly", "whimsical"],
    6: ["bright", "cheerful", "gay", "happy", "joyous", "merry"],
    7: ["agitated", "dramatic", "exciting", "exhilarated", "impetuous", "passionate", "restless", "sensational", "soaring", "triumphant"],
    8: ["emphatic", "exalting", "majestic", "martial", "ponderous", "robust", "vigorous"]
}

# Load the emotion labels
with open("emotion_labels_texts.json", "r") as f:
    emotion_labels = json.load(f)

# Map emotions to clusters
mapped_labels = {}
for idx, label in emotion_labels.items():
    label_lower = label.lower()
    found_cluster = None
    for cluster_id, keywords in mood_clusters.items():
        if any(re.search(rf"\b{kw}\b", label_lower) for kw in keywords):
            found_cluster = cluster_id
            break
    mapped_labels[idx] = found_cluster if found_cluster is not None else "unknown"

# Save mapped results
with open("emotion_cluster_mapped.json", "w") as f:
    json.dump(mapped_labels, f, indent=2)

print("Done: mapped emotions saved to emotion_cluster_mapped.json")


