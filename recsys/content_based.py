"""
Content-based filtering using item metadata (genres + tags) and cosine
similarity, with a linear calibration step mapping similarity onto the
1-5 rating scale.

--- The math ---

Step 1 - item vectors (TF-IDF):
Each item's genres and tags are concatenated into a small "document"
(e.g. "Action Sci-Fi cult-favorite twist-ending"). We vectorize every
item's document with TF-IDF (term frequency - inverse document
frequency): each dimension is one genre/tag "term", and its value is
high when that term appears in the item's document and is relatively
rare across the whole catalog (i.e. distinctive), and low when the term
is generic (appears on almost every item, e.g. an overused tag). This
gives every item a vector in "genre/tag space" where items that share
more, and more distinctive, genres/tags sit closer together. Rows are
L2-normalized so the vectors' magnitude doesn't matter, only direction.

Step 2 - user profile vectors:
A user's taste profile is built as a *rating-weighted average* of the
TF-IDF vectors of the items they've rated in the training set, using
ratings centered on the global mean rating (rating - mu) as weights:

    profile[u] = sum_i (r_ui - mu) * item_vector[i]   over items i rated by u

Centering matters: an item the user rated 5 (well above average) pulls
the profile *toward* its genres/tags, while an item they rated 1 (well
below average) pushes the profile *away* from its genres/tags (negative
weight). An item rated close to the user's usual average contributes
almost nothing, since it isn't informative about their taste (everyone
roughly agrees on it). The resulting profile is then L2-normalized.

Step 3 - scoring via cosine similarity:
Since both user profiles and item vectors are unit-normalized, their dot
product equals the cosine of the angle between them - a similarity in
[-1, 1] that is high when the user's profile points in the same
"genre/tag direction" as the item.

Step 4 - calibration onto the rating scale:
Cosine similarity isn't naturally on a 1-5 scale, so we fit a simple 1-D
linear regression (least squares) mapping similarity -> observed training
rating: rating_hat = a + b * similarity. This lets content-based
predictions be compared apples-to-apples against collaborative filtering
on RMSE/MAE, and it's a defensible, from-scratch calibration step (not
just an arbitrary rescale).
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def _build_item_documents(items_df):
    """Turn 'genres' and 'tags' pipe-separated columns into a bag-of-terms doc string."""
    genres = items_df["genres"].fillna("").str.replace("|", " ", regex=False)
    tags = items_df["tags"].fillna("").str.replace("|", " ", regex=False)
    return (genres + " " + tags).str.strip()


class ContentBasedRecommender:
    def __init__(self):
        self.vectorizer = None
        self.item_matrix = None       # n_items x n_features, L2-normalized rows
        self.user_profiles = None     # n_users x n_features, L2-normalized rows
        self.global_mean = 0.0
        self.calib_a = 0.0
        self.calib_b = 0.0
        self.n_users = None
        self.n_items = None

    def fit(self, train_ratings, items_df, user_map, item_map):
        self.n_users = len(user_map)
        self.n_items = len(item_map)
        self.global_mean = float(train_ratings["rating"].mean())

        # Order items_df rows to match item_map's dense index order.
        items_ordered = items_df.set_index("item_id").loc[item_map.ids].reset_index()
        docs = _build_item_documents(items_ordered)

        self.vectorizer = TfidfVectorizer(token_pattern=r"[^\s]+")
        self.item_matrix = self.vectorizer.fit_transform(docs).toarray()  # already L2-normalized rows

        # Build the user rating-weight matrix (n_users x n_items), centered on global mean.
        u_idx = user_map.to_idx_array(train_ratings["user_id"].values)
        i_idx = item_map.to_idx_array(train_ratings["item_id"].values)
        centered = train_ratings["rating"].values.astype(np.float64) - self.global_mean

        W = np.zeros((self.n_users, self.n_items), dtype=np.float64)
        W[u_idx, i_idx] = centered

        profiles = W.dot(self.item_matrix)  # n_users x n_features
        norms = np.linalg.norm(profiles, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid div-by-zero for cold-start users with no ratings
        self.user_profiles = profiles / norms

        # Calibrate similarity -> rating via least-squares linear fit on training pairs.
        sims = np.sum(self.user_profiles[u_idx] * self.item_matrix[i_idx], axis=1)
        ratings = train_ratings["rating"].values.astype(np.float64)
        A = np.vstack([sims, np.ones_like(sims)]).T
        (b, a), *_ = np.linalg.lstsq(A, ratings, rcond=None)
        self.calib_a, self.calib_b = float(a), float(b)

        return self

    def _similarity(self, u):
        if u < 0 or u >= self.n_users:
            return np.zeros(self.n_items)
        return self.item_matrix.dot(self.user_profiles[u])

    def predict(self, u, i):
        if u < 0 or u >= self.n_users or i < 0 or i >= self.n_items:
            return self.global_mean
        sim = float(self.item_matrix[i].dot(self.user_profiles[u]))
        return float(np.clip(self.calib_a + self.calib_b * sim, 1.0, 5.0))

    def predict_all_for_user(self, u):
        """Vectorized calibrated score for every item (unclipped, for ranking)."""
        sims = self._similarity(u)
        return self.calib_a + self.calib_b * sims

    def get_similar_items(self, item_idx, top_k=10):
        """Utility: items most similar to a given item by content (not user-specific)."""
        sims = self.item_matrix.dot(self.item_matrix[item_idx])
        order = np.argsort(-sims)
        order = order[order != item_idx][:top_k]
        return order, sims[order]
