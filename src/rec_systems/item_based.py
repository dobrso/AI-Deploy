from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ItemBasedRec:
    def __init__(self, df, train_likes):
        self.n_items = len(df)
        n_users = len(train_likes)
        self.user_ids = sorted(train_likes.keys())
        self.uid2row = {u: i for i, u in enumerate(self.user_ids)}

        R = np.zeros((self.n_items, n_users), dtype=np.float32)
        for u, items in train_likes.items():
            for i in items:
                R[i, self.uid2row[u]] = 1.0
        self.item_sim = cosine_similarity(R)

    def recommend(self, user_id, train_items, k=10, k_similar=20):
        seen = set(train_items)
        scores = defaultdict(float)
        sim_sums = defaultdict(float)

        for i in train_items:
            sims = self.item_sim[i].copy()
            sims[i] = -np.inf
            top_sim = np.argsort(-sims)[:k_similar]
            for j in top_sim:
                if j in seen:
                    continue
                scores[j] += sims[j]
                sim_sums[j] += sims[j]

        ranked = sorted(
            [(j, scores[j] / sim_sums[j]) for j in scores if sim_sums[j] > 0],
            key=lambda x: x[1], reverse=True
        )[:k]
        return [j for j, _ in ranked]