# Lyric Search

Depolyed at https://ernestchu.github.io/lyric-search/  
Frontend: https://ernestchu.github.io/lyric-search/  
Backend: https://huggingface.co/spaces/ernestchu/lyric-search

## TODO
- [x] longest common subsequence
- [x] Scroll to top after search
- [x] add clear search bar button
- [x] Randomly sample 10,000 songs. For each song, randomly select a snippet of lyrics. Then, use our app to search with the snippet and evaluate whether the correct song appears within the top-k ranked search results.
- [x] Add [Latent Semantic Indexing](https://radimrehurek.com/gensim/auto_examples/core/run_topics_and_transformations.html)
- [ ] term frequency upper bound

| USE_LSI | USE_LCS | Top-1 | Top-5 | Top-10 | Top-20 | Top-50 |
|---------|---------|-------|-------|--------|--------|--------|
| ❌      | ❌      | 0.136 | 0.293 | 0.359  | 0.428  | 0.524  |
| ❌      | ✅      | 0.146 | 0.353 | 0.462  | 0.545  | 0.642  |
| ✅      | ❌      | 0.000 | 0.000 | 0.000  | 0.000  | 0.000  |
| ✅      | ✅      | 0.000 | 0.000 | 0.000  | 0.000  | 0.000  |

