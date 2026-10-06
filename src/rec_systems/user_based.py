from collections import defaultdict

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class UserBasedRec:
    def __init__(self, df, train_likes):
        self.n_items = len(df)
        n_users = len(train_likes)
        self.user_ids = sorted(train_likes.keys())
        self.uid2row = {u: i for i, u in enumerate(self.user_ids)}

        R = np.zeros((n_users, self.n_items), dtype=np.float32)
        for u, items in train_likes.items():
            for i in items:
                R[self.uid2row[u], i] = 1.0
        self.R = R
        self.user_sim = cosine_similarity(R)

    def recommend(self, user_id, train_items, k=10, k_neighbors=20):
        if user_id not in self.uid2row:
            return []
        urow = self.uid2row[user_id]
        sims = self.user_sim[urow].copy()
        sims[urow] = -np.inf
        neighbors = np.argsort(-sims)[:k_neighbors]

        scores = defaultdict(float)
        sim_sums = defaultdict(float)
        seen = train_items

        for n in neighbors:
            for i in np.where(self.R[n] > 0)[0]:
                if i in seen:
                    continue
                scores[i] += sims[n]
                sim_sums[i] += sims[n]

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        return [i for i, _ in ranked]