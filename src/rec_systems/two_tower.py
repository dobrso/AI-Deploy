import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy.sparse import csr_matrix


SEED = 42
AUDIO_COLS = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence",
    "tempo", "duration_ms",
]

class TwoTower(nn.Module):
    def __init__(self, n_items, content_dim, dim=64, use_id=True):
        super().__init__()
        self.use_id = use_id
        self.use_content = content_dim > 0

        if use_id:
            self.item_id = nn.Embedding(n_items, dim)
            nn.init.normal_(self.item_id.weight, std=0.1)
        if self.use_content:
            self.item_mlp = nn.Sequential(
                nn.Linear(content_dim, 256), nn.ReLU(), nn.Linear(256, dim))

        self.user_mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim))

    def item_vectors(self, content, cold_items=None):
        v = 0
        if self.use_id:
            w = self.item_id.weight
            if cold_items is not None:
                w = w.clone()
                w[cold_items] = 0
            v = v + w
        if self.use_content:
            v = v + self.item_mlp(content)
        return F.normalize(v, dim=-1)

    def user_vectors(self, hist, item_vecs):
        h = hist @ item_vecs
        return F.normalize(h + self.user_mlp(h), dim=-1)

def train_two_tower(X, content=None, use_id=True, dim=64, temp=0.05,
                    epochs=15, batch_size=512, lr=1e-3, cold_items=None):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)

    Xb = (X > 0).astype(np.float32)
    n_users, n_items = Xb.shape
    H_all = torch.tensor(Xb.toarray()) if hasattr(Xb, "toarray") \
            else torch.tensor(np.asarray(Xb))
    C = torch.tensor(content, dtype=torch.float32) if content is not None \
        else torch.zeros(n_items, 0)

    not_in_catalog = torch.zeros(n_items, dtype=torch.bool)
    if cold_items is not None:
        not_in_catalog[torch.as_tensor(cold_items)] = True

    model = TwoTower(n_items, C.shape[1], dim, use_id)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    u_idx, i_idx = np.nonzero(Xb)

    for epoch in range(epochs):
        t0 = time.time()
        perm = rng.permutation(len(u_idx))
        total = 0.0
        for start in range(0, len(perm), batch_size):
            b = perm[start:start + batch_size]
            u = torch.as_tensor(u_idx[b], dtype=torch.long)
            i = torch.as_tensor(i_idx[b], dtype=torch.long)
            rows = torch.arange(len(b))

            hist = H_all[u].clone()
            hist[rows, i] = 0
            other_pos = hist > 0
            hist = hist / hist.sum(1, keepdim=True).clamp(min=1)

            I = model.item_vectors(C)
            U = model.user_vectors(hist, I)
            logits = (U @ I.T) / temp
            logits = logits.masked_fill(other_pos | not_in_catalog, -1e9)
            loss = F.cross_entropy(logits, i)

            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(b)

    model.eval()
    with torch.no_grad():
        I = model.item_vectors(C, cold_items)
        hist = H_all / H_all.sum(1, keepdim=True).clamp(min=1)
        U = model.user_vectors(hist, I)
    return U.numpy(), I.numpy()

class TwoTowerRec:
    def __init__(self, df, train_likes, dim=64, epochs=10, temp=0.05,
                 use_id=True, use_content=True, batch_size=512):
        self.df = df.reset_index(drop=True)
        self.n_items = len(self.df)
        self.user_ids = sorted(train_likes.keys())
        self.uid2idx = {u: k for k, u in enumerate(self.user_ids)}

        rows, cols = [], []
        for u, items in train_likes.items():
            r = self.uid2idx[u]
            for i in items:
                rows.append(r); cols.append(i)
        X = csr_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)),
            shape=(len(self.user_ids), self.n_items))

        content = None
        if use_content:
            audio = StandardScaler().fit_transform(
                self.df[AUDIO_COLS].values.astype(np.float32)).astype(np.float32)
            genres = self.df["track_genre"].astype(str).values
            genc = LabelEncoder().fit(genres)
            g_onehot = np.eye(len(genc.classes_), dtype=np.float32)[
                genc.transform(genres)]
            content = np.concatenate([audio, g_onehot], axis=1)

        U, I = train_two_tower(
            X, content=content, use_id=use_id, dim=dim, temp=temp,
            epochs=epochs, batch_size=batch_size)

        self.U = U
        self.I = I

    def recommend(self, user_id, train_items, k=10):
        if user_id in self.uid2idx:
            u_vec = self.U[self.uid2idx[user_id]]
        else:
            if not train_items:
                return []
            idxs = np.array(list(train_items))
            u_vec = self.I[idxs].mean(axis=0)
            n = np.linalg.norm(u_vec)
            if n > 0:
                u_vec = u_vec / n

        scores = self.I @ u_vec
        seen = set(train_items)
        order = np.argsort(-scores)
        out = [int(i) for i in order if int(i) not in seen][:k]
        return out