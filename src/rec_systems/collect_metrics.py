import glob
import os
import time

import kagglehub
import pandas as pd
import numpy as np

from content_based import ContentBasedRec
from user_based import UserBasedRec
from item_based import ItemBasedRec
from popularity_based import PopularityBasedRec

def load_spotify_dataset(sample_n=None, seed=42):
    dataset_name = 'maharshipandya/-spotify-tracks-dataset'
    path = kagglehub.dataset_download(dataset_name)
    csv_files = glob.glob(os.path.join(path, '**', '*.csv'), recursive=True)

    df = pd.read_csv(csv_files[0])
    if 'Unnamed: 0' in df.columns:
        df = df.drop(columns=['Unnamed: 0'])
    df = df.drop_duplicates(subset='track_id').reset_index(drop=True)
    df['artists'] = df['artists'].fillna('unknown')
    df['track_genre'] = df['track_genre'].fillna('unknown')
    df['track_name'] = df['track_name'].fillna('unknown')

    if sample_n is not None and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=seed).reset_index(drop=True)

    return df

def simulate_users(df, n_users=1000, likes_per_user=100,
                              test_ratio=0.2, n_genres_per_user=2, seed=42):
    rng = np.random.default_rng(seed)
    genres = df['track_genre'].values
    unique_genres = df['track_genre'].unique()

    genre_to_idx = {g: np.where(genres == g)[0] for g in unique_genres}

    train_likes, test_likes = {}, {}

    for u in range(n_users):
        n_g = rng.integers(1, n_genres_per_user + 1)
        user_genres = rng.choice(unique_genres, n_g, replace=False)

        pool = np.concatenate([genre_to_idx[g] for g in user_genres])
        if len(pool) < likes_per_user:
            pool = np.arange(len(df))

        idxs = rng.choice(pool, likes_per_user, replace=False)
        n_test = max(1, int(likes_per_user * test_ratio))
        test_likes[u]  = set(idxs[:n_test].tolist())
        train_likes[u] = set(idxs[n_test:].tolist())

    return train_likes, test_likes

def precision_at_k(rec, rel, k):
    return len(set(rec[:k]) & set(rel)) / k if k else 0.0

def recall_at_k(rec, rel, k):
    return len(set(rec[:k]) & set(rel)) / len(rel) if rel else 0.0

def hit_rate_at_k(rec, rel, k):
    return 1.0 if len(set(rec[:k]) & set(rel)) > 0 else 0.0

def ndcg_at_k(rec, rel, k):
    rel_set = set(rel)
    dcg = sum(1.0 / np.log2(i + 2) for i, x in enumerate(rec[:k]) if x in rel_set)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel_set), k)))
    return dcg / idcg if idcg > 0 else 0.0


def evaluate_recommender(recommend_fn, train_likes, test_likes, k=10):
    precisions, recalls, hits, ndcgs = [], [], [], []
    t0 = time.time()

    for u, train_items in train_likes.items():
        if u not in test_likes or not test_likes[u]:
            continue
        rec = recommend_fn(u, train_items, k)
        rel = test_likes[u]

        precisions.append(precision_at_k(rec, rel, k))
        recalls.append(recall_at_k(rec, rel, k))
        hits.append(hit_rate_at_k(rec, rel, k))
        ndcgs.append(ndcg_at_k(rec, rel, k))

    elapsed = time.time() - t0
    return {
        f'Precision@{k}': round(float(np.mean(precisions)), 4),
        f'Recall@{k}': round(float(np.mean(recalls)), 4),
        f'HitRate@{k}': round(float(np.mean(hits)), 4),
        f'NDCG@{k}': round(float(np.mean(ndcgs)), 4),
        'Время работы': round(elapsed, 2),
    }

def main():
    K = 10
    df = load_spotify_dataset(sample_n=20000)

    print('=' * 60)

    train_likes, test_likes = simulate_users(df, n_users=1000, likes_per_user=100)
    print(f'Mock-пользователей: {len(train_likes)}')
    print(f'Train лайков на пользователя: ~{np.mean([len(v) for v in train_likes.values()]):.1f}')
    print(f'Test лайков на пользователя:  ~{np.mean([len(v) for v in test_likes.values()]):.1f}')

    t0 = time.time()
    cbr = ContentBasedRec(df)
    print(f'ContentBased построен за {time.time() - t0:.2f} с')

    t0 = time.time()
    ubr = UserBasedRec(df, train_likes)
    print(f'UserBased построен за {time.time() - t0:.2f} с')

    t0 = time.time()
    ibr = ItemBasedRec(df, train_likes)
    print(f'ItemBased построен за {time.time() - t0:.2f} с')

    t0 = time.time()
    pbr = PopularityBasedRec(df)
    print(f'PopularityBased построен за {time.time() - t0:.2f} с')

    results = {}

    results['ContentBased'] = evaluate_recommender(
        lambda u, t, k: cbr.recommend(u, t, k), train_likes, test_likes, K
    )

    results['UserBased'] = evaluate_recommender(
        lambda u, t, k: ubr.recommend(u, t, k), train_likes, test_likes, K
    )

    results['ItemBased'] = evaluate_recommender(
        lambda u, t, k: ibr.recommend(u, t, k), train_likes, test_likes, K
    )

    results['PopularityBased'] = evaluate_recommender(
        lambda u, t, k: pbr.recommend(u, t, k), train_likes, test_likes, K
    )

    print('=' * 60)
    print(f'СВОДНАЯ ТАБЛИЦА {K} МЕТРИК')
    print('=' * 60)
    summary = pd.DataFrame(results).T
    print(summary.to_string())


if __name__ == "__main__":
    main()