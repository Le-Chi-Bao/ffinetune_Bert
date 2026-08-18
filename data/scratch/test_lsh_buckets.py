import time
import pandas as pd
from datasketch import MinHash, MinHashLSH

train_df = pd.read_csv("raw/train.csv")
titles = train_df["Title"].fillna("").astype(str).tolist()

print(f"Testing ultra-fast MinHash LSH bucket extraction on {len(titles)} items...")
t0 = time.time()
num_perm = 32
lsh = MinHashLSH(threshold=0.90, num_perm=num_perm)

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
print(f"Generated {len(minhashes)} MinHashes in {t1 - t0:.2f}s")

with lsh.insertion_session() as session:
    for r_id, m in minhashes.items():
        session.insert(r_id, m)

t2 = time.time()
print(f"Inserted into LSH in {t2 - t1:.2f}s")

# Ultra-fast bucket candidate extraction
candidate_pairs = set()
for hashtable in lsh.hashtables:
    for bucket_key in hashtable:
        bucket = hashtable[bucket_key]
        if len(bucket) > 1:
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    u, v = bucket[i], bucket[j]
                    if u > v:
                        u, v = v, u
                    candidate_pairs.add((u, v))

t3 = time.time()
print(f"Extracted {len(candidate_pairs)} candidate pairs from LSH buckets in {t3 - t2:.2f}s!")

# Verify Jaccard similarity for candidate pairs
verified_pairs = 0
for u, v in candidate_pairs:
    sim = minhashes[u].jaccard(minhashes[v])
    if sim >= 0.90:
        verified_pairs += 1

t4 = time.time()
print(f"Verified Jaccard similarity in {t4 - t3:.2f}s. Found {verified_pairs} near-duplicate pairs!")
