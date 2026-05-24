"""
Sanity checks for the matrix-factorization training loop: does it run,
does its training loss actually decrease, and does it overfit a tiny toy
matrix it should easily be able to memorize.
"""
import sys
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recsys.collaborative import MatrixFactorization


def _toy_ratings():
    """
    A small, fully-specified 4-user x 5-item rating set with a clear
    low-rank structure (two "taste clusters") that matrix factorization
    with a handful of latent factors should be able to fit closely.
    """
    rng = np.random.default_rng(0)
    n_users, n_items = 4, 5
    # user 0,1 like items 0,1,2 ; user 2,3 like items 2,3,4
    base = np.array([
        [5, 4, 3, 2, 1],
        [4, 5, 3, 1, 2],
        [2, 1, 3, 5, 4],
        [1, 2, 3, 4, 5],
    ], dtype=np.float64)
    u_idx, i_idx, ratings = [], [], []
    for u in range(n_users):
        for i in range(n_items):
            u_idx.append(u)
            i_idx.append(i)
            ratings.append(base[u, i])
    return (np.array(u_idx), np.array(i_idx), np.array(ratings), n_users, n_items)


def test_training_loss_decreases_over_epochs():
    u_idx, i_idx, ratings, n_users, n_items = _toy_ratings()
    model = MatrixFactorization(n_factors=3, lr=0.05, reg=0.02, n_epochs=40, seed=1)
    model.fit(u_idx, i_idx, ratings, n_users=n_users, n_items=n_items)

    history = model.train_rmse_history
    assert len(history) == 40
    # Loss should be (near-)monotonically decreasing: compare early vs. late
    # average rather than requiring every single epoch to improve (SGD is
    # stochastic and can have small bumps), which is still a strong and
    # meaningful check.
    early_avg = np.mean(history[:5])
    late_avg = np.mean(history[-5:])
    assert late_avg < early_avg, f"expected loss to decrease: early={early_avg}, late={late_avg}"
    # Final loss should be substantially better than the first epoch's.
    assert history[-1] < history[0] * 0.8


def test_model_can_fit_a_small_low_rank_matrix_closely():
    u_idx, i_idx, ratings, n_users, n_items = _toy_ratings()
    model = MatrixFactorization(n_factors=4, lr=0.05, reg=0.005, n_epochs=300, seed=1)
    model.fit(u_idx, i_idx, ratings, n_users=n_users, n_items=n_items)

    preds = np.array([model.predict(u, i) for u, i in zip(u_idx, i_idx)])
    train_rmse = np.sqrt(np.mean((preds - ratings) ** 2))
    # With enough factors, low regularization, and enough epochs, the model
    # should be able to nearly memorize a fully-observed 4x5 matrix.
    assert train_rmse < 0.5


def test_predictions_are_clipped_to_valid_rating_range():
    u_idx, i_idx, ratings, n_users, n_items = _toy_ratings()
    model = MatrixFactorization(n_factors=3, lr=0.05, reg=0.02, n_epochs=20, seed=1)
    model.fit(u_idx, i_idx, ratings, n_users=n_users, n_items=n_items)
    for u in range(n_users):
        for i in range(n_items):
            p = model.predict(u, i)
            assert 1.0 <= p <= 5.0


def test_predict_all_for_user_matches_individual_predict_order():
    u_idx, i_idx, ratings, n_users, n_items = _toy_ratings()
    model = MatrixFactorization(n_factors=3, lr=0.05, reg=0.02, n_epochs=20, seed=1)
    model.fit(u_idx, i_idx, ratings, n_users=n_users, n_items=n_items)

    all_scores = model.predict_all_for_user(0)
    assert len(all_scores) == n_items
    # predict_all_for_user is unclipped; individual predict() is clipped to
    # [1, 5]. They should still agree once we apply the same clipping.
    clipped = np.clip(all_scores, 1.0, 5.0)
    for i in range(n_items):
        assert abs(clipped[i] - model.predict(0, i)) < 1e-6


def test_unknown_user_or_item_falls_back_to_global_mean():
    u_idx, i_idx, ratings, n_users, n_items = _toy_ratings()
    model = MatrixFactorization(n_factors=3, lr=0.05, reg=0.02, n_epochs=10, seed=1)
    model.fit(u_idx, i_idx, ratings, n_users=n_users, n_items=n_items)
    assert model.predict(n_users + 10, 0) == model.global_mean
    assert model.predict(0, n_items + 10) == model.global_mean
