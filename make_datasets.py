import sqlite3
from datasets import Dataset

preview = False
preview_num_samples = 10000

def get_all_tracks_with_lyrics():
    # Download the db at https://lrclib.net/db-dumps
    conn = sqlite3.connect('lrclib-db-dump-20250319T053125Z.sqlite3')
    conn.row_factory = sqlite3.Row  # Enable row factory for dictionary-like access
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM tracks{' LIMIT ' + str(preview_num_samples) if preview else ''};")
    for track in cursor:
        track = dict(track)
        lyrics_id = track['last_lyrics_id']
        c = conn.cursor()
        c.execute(f"SELECT * FROM lyrics WHERE id == {lyrics_id};")
        lyrics = dict(c.fetchone())
        for key, value in lyrics.items():
            if key in ['id', 'track_id']:
                continue
            if key in ['created_at', 'updated_at']:
                track[f'lyrics_{key}'] = value
            track[key] = value
        yield track

ds = Dataset.from_generator(get_all_tracks_with_lyrics)
if preview:
    ds.push_to_hub('ernestchu/lrclib-20250319', 'preview')
else:
    ds.push_to_hub('ernestchu/lrclib-20250319')

