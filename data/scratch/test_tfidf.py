import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

train_df = pd.read_csv("raw/train.csv")
canonical_texts = (train_df["Title"].fillna("") + "\n" + train_df["Description"].fillna("")).str.casefold().tolist()

print(f"Testing TF-IDF Char N-Gram (3-5) on {len(canonical_texts)} samples...")
t0 = time.time()
vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 5), min_df=2)
X = vectorizer.fit_transform(canonical_texts)
t1 = time.time()
print(f"Vectorized shape: {X.shape} in {t1 - t0:.2f} s")

# Fast threshold search using NearestNeighbors
nn = NearestNeighbors(n_neighbors=5, metric='cosine', algorithm='brute', n_jobs=-1)
nn.fit(X)
t2 = time.time()
print(f"Fitted NearestNeighbors in {t2 - t1:.2f} s")

# Cosine distance = 1 - Cosine similarity
# Threshold similarity 0.90 => radius = 1 - 0.90 = 0.10
distances, indices = nn.radius_neighbors(X, radius=0.101)
t3 = time.time()
print(f"Radius query finished in {t3 - t2:.2f} s")

near_dup_count = 0
for i, (dists, idxs) in enumerate(zip(distances, indices)):
    for d, j in zip(dists, idxs):
        if i < j: # pair (i, j)
            near_dup_count += 1
print(f"Found {near_dup_count} near-duplicate pairs (similarity >= 0.90)")
