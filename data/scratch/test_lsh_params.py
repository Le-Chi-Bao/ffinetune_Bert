import time
import pandas as pd
from datasketch import MinHash, MinHashLSH

train_df = pd.read_csv("raw/train.csv")
titles = train_df["Title"].fillna("").astype(str).tolist()

print(f"Testing tuned MinHashLSH on {len(titles)} items...", flush=True)
t0 = time.time()
num_perm = 128
lsh = MinHashLSH(threshold=0.90, num_perm=num_perm, weights=(0.5, 0.5))
print(f"LSH b={lsh.b}, r={lsh.r}", flush=True)

def get_ngrams(text, n=3):
    text = text.lower()
    return set(text[i:i+n] for i in range(len(text)-n+1))

minhashes = {}
for idx, text in enumerate(titles):
    m = MinHash(num_perm=num_perm)
    for ngram in get_ngrams(text):
        m.update(ngram.encode('utf-8'))
    minhashes[f"id_{idx}"] = m

t1 = time.time()
print(f"Generated {len(minhashes)} MinHashes in {t1 - t0:.2f}s", flush=True)

with lsh.insertion_session() as session:
    for r_id, m in minhashes.items():
        session.insert(r_id, m)

t2 = time.time()
print(f"Inserted into LSH in {t2 - t1:.2f}s", flush=True)

candidate_pairs = set()
for r_id, m in minhashes.items():
    res = lsh.query(m)
    for other_id in res:
        if r_id < other_id:
            candidate_pairs.add((r_id, other_id))

t3 = time.time()
print(f"Queried LSH in {t3 - t2:.2f}s. Candidate pairs: {len(candidate_pairs)}", flush=True)

verified = 0
for u, v in candidate_pairs:
    if minhashes[u].jaccard(minhashes[v]) >= 0.90:
        verified += 1
t4 = time.time()
print(f"Verified Jaccard in {t4 - t3:.2f}s. Near-duplicate pairs (>=0.90): {verified}", flush=True)
