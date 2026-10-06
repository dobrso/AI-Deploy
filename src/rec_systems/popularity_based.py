import numpy as np

class PopularityBasedRec:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        pop = self.df['popularity'].values.astype(float)
        self.pop_norm = (pop - pop.min()) / (pop.max() - pop.min() + 1e-9)
        self.ranked = np.argsort(-self.pop_norm)

    def recommend(self, user_id, train_items, k=10):
        seen = set(train_items)
        out = []
        for i in self.ranked:
            if i in seen:
                continue
            out.append(int(i))
            if len(out) == k:
                break
        return out