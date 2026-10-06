import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRec:
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        text = (self.df['track_genre'].astype(str) + ' ' +
                self.df['artists'].astype(str).str.replace(';', ' ', regex=False))
        tfidf = TfidfVectorizer(token_pattern=r"[^\s]+")
        self.matrix = tfidf.fit_transform(text)
        self.n_items = self.matrix.shape[0]

    def recommend(self, user_id, train_items, k=10):
        idxs = list(train_items)
        if not idxs:
            return []
        profile = np.asarray(self.matrix[idxs].mean(axis=0))
        sims = cosine_similarity(profile, self.matrix).ravel()
        for i in idxs:
            sims[i] = -np.inf
        return np.argsort(-sims)[:k].tolist()