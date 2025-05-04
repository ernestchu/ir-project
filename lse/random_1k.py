from datasets import load_dataset, Dataset

NUM_EXAMPLES = 1000
SEED = 20250430

def random_1k():
    count = 0
    for row in load_dataset("ernestchu/lrclib-20250319", split='train').shuffle(seed=SEED):
        sub = {k: row[k] for k in ['id', 'plain_lyrics']}
        if count >= NUM_EXAMPLES:
            break
        if sub['plain_lyrics'] and len(sub['plain_lyrics']) > 10:
            count += 1
            yield sub

ds = Dataset.from_generator(random_1k)
ds.push_to_hub('ernestchu/lrclib-20250319', 'random-1k')
