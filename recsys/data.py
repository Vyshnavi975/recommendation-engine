"""
Data loading, ID indexing, and train/test splitting utilities.

All models internally work with dense zero-based matrix indices rather
than raw user_id/item_id values, so this module is the single place that
maps between the two (via `IdMap`) and is shared by every model so their
index spaces stay consistent.
"""
import os
import numpy as np
import pandas as pd


DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


class IdMap:
    """Bidirectional mapping between raw ids (e.g. user_id) and dense 0..N-1 indices."""

    def __init__(self, ids):
        self.ids = np.array(sorted(pd.unique(ids)))
        self.id_to_idx = {raw_id: i for i, raw_id in enumerate(self.ids)}

    def __len__(self):
        return len(self.ids)

    def to_idx(self, raw_id):
        return self.id_to_idx[raw_id]

    def to_idx_array(self, raw_ids):
        return np.array([self.id_to_idx[r] for r in raw_ids])

    def to_raw(self, idx):
        return self.ids[idx]

    def contains(self, raw_id):
        return raw_id in self.id_to_idx


def load_data(data_dir=DEFAULT_DATA_DIR):
    """Load ratings, items, and users CSVs into DataFrames."""
    ratings = pd.read_csv(os.path.join(data_dir, "ratings.csv"))
    items = pd.read_csv(os.path.join(data_dir, "items.csv"))
    users = pd.read_csv(os.path.join(data_dir, "users.csv"))
    return ratings, items, users


def train_test_split_per_user(ratings, test_frac=0.2, min_ratings=5, seed=42):
    """
    Per-user random holdout split.

    For each user with at least `min_ratings` ratings, a random `test_frac`
    fraction of their ratings is held out as test data (at least 1 rating,
    capped so at least 1 remains in train). Users with fewer than
    `min_ratings` ratings have all of their ratings kept in train only,
    since there isn't enough signal to fairly evaluate on them and
    removing their only ratings would make it impossible to learn their
    preferences at all.

    This "per-user holdout" scheme (rather than a single global random
    split) matters for recommender evaluation: it guarantees every test
    user has some training history to build a profile from, which mirrors
    how the model would actually be used in production (predict next
    ratings for existing users), and it prevents users from disappearing
    entirely from the training set.
    """
    rng = np.random.default_rng(seed)
    train_parts = []
    test_parts = []

    for user_id, group in ratings.groupby("user_id"):
        n = len(group)
        if n < min_ratings:
            train_parts.append(group)
            continue
        n_test = max(1, int(round(n * test_frac)))
        n_test = min(n_test, n - 1)  # always leave >=1 rating in train
        idx = rng.permutation(n)
        test_idx = idx[:n_test]
        train_idx = idx[n_test:]
        group_arr = group.reset_index(drop=True)
        test_parts.append(group_arr.iloc[test_idx])
        train_parts.append(group_arr.iloc[train_idx])

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else ratings.iloc[0:0].copy()
    return train_df, test_df


def build_user_item_matrix(ratings, user_map, item_map):
    """Build a dense |users| x |items| matrix with NaN for missing entries."""
    mat = np.full((len(user_map), len(item_map)), np.nan, dtype=np.float32)
    u_idx = user_map.to_idx_array(ratings["user_id"].values)
    i_idx = item_map.to_idx_array(ratings["item_id"].values)
    mat[u_idx, i_idx] = ratings["rating"].values
    return mat


def get_user_rated_items(ratings, user_id):
    """Set of item_ids a user has already rated (used to exclude from recommendations)."""
    return set(ratings.loc[ratings["user_id"] == user_id, "item_id"].values)
