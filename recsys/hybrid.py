"""
Hybrid recommender: a weighted linear blend of the collaborative-filtering
(matrix factorization) score and the content-based (TF-IDF cosine
similarity) score.

    hybrid_score(u, i) = alpha * cf_score(u, i) + (1 - alpha) * content_score(u, i)

Why blend at all? The two methods fail in complementary ways:
  - Collaborative filtering is powerful once a user/item has enough rating
    history (it can capture taste patterns no metadata field would
    encode), but degrades for new/sparsely-rated items and users
    ("cold start") since it has nothing to learn from.
  - Content-based filtering works from day one for a new item (as long as
    it has genre/tag metadata) and is easy to explain ("recommended
    because you liked other Sci-Fi movies"), but is limited to whatever
    the metadata captures and can't discover cross-genre taste patterns
    collaborative filtering finds naturally.

A simple convex combination (alpha in [0, 1], weights summing to 1) lets
us dial between the two. Both `predict()` outputs are already calibrated
to the 1-5 rating scale, so the weighted sum stays in that range too.
alpha is a hyperparameter; `evaluation.py`'s CLI reports metrics at a
default alpha, and it can be tuned by comparing validation RMSE at a few
candidate values (see README for a swept comparison).
"""
import numpy as np


class HybridRecommender:
    def __init__(self, cf_model, cb_model, alpha=0.7):
        """
        alpha: weight on the collaborative-filtering score. (1 - alpha) is
        the weight on the content-based score. alpha=1.0 reduces to pure
        CF; alpha=0.0 reduces to pure content-based.
        """
        self.cf = cf_model
        self.cb = cb_model
        self.alpha = alpha

    def predict(self, u, i):
        cf_score = self.cf.predict(u, i)
        cb_score = self.cb.predict(u, i)
        blended = self.alpha * cf_score + (1 - self.alpha) * cb_score
        return float(np.clip(blended, 1.0, 5.0))

    def predict_all_for_user(self, u):
        cf_scores = self.cf.predict_all_for_user(u)
        cb_scores = self.cb.predict_all_for_user(u)
        return self.alpha * cf_scores + (1 - self.alpha) * cb_scores
