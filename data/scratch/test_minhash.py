import time
import pandas as pd
from datasketch import MinHash, MinHashLSH

train_df = pd.read_csv("raw/train.csv")
titles = train_df["Title"].fillna("").astype(str).tolist()

print("Testing MinHash on 10,000 samples...")
t0 = time.time()
lsh = MinHashLSH(threshold=0.90, num_perm=64)

def get_ngrams(text, n=3):
    text = text.lower()
    return set(text[i:i+n] for i in range(len(text)-n+1))

minhashes = []
for idx, text in enumerate(titles[:10000]):
    m = MinHash(num_perm=64)
    for ngram in get_ngrams(text):
        m.update(ngram.encode('utf-8'))
    lsh.insert(f"id_{idx}", m)
    minhashes.append(m)

t1 = time.time()
print(f"10,000 items processed in {t1 - t0:.2f} seconds.")
