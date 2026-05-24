"""
Offline evaluation metrics, implemented from scratch (no scikit-learn
metric functions), plus a harness that runs rating-prediction and
ranking evaluation for any model exposing `predict(u_idx, i_idx)` and
`predict_all_for_user(u_idx)`.

Two families of metrics are computed, because they answer different
questions:

  1. Rating-prediction accuracy (RMSE, MAE): "if I ask the model to
     predict the exact star rating a user would give an item, how close
     is it?" This is the classic Netflix-Prize-style metric.

  2. Top-N ranking quality (Precision@K, Recall@K, NDCG@K): "if I show
     the user a ranked list of K items they haven't seen, how many are
     actually ones they'd like?" This is closer to how a recommender is
     actually used in a product (a ranked list, not a predicted score),
     and a model can do well on RMSE while doing poorly here (e.g. by
     being very good at predicting "meh" ratings accurately but bad at
     telling great items apart from good ones).
"""
import numpy as np


# ----------------------------- rating metrics ----------------------------- #

def rmse(y_true, y_pred):
    """Root Mean Squared Error: sqrt(mean((true - pred)^2)). Penalizes large errors more."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true, y_pred):
    """Mean Absolute Error: mean(|true - pred|). More interpretable / robust to outliers than RMSE."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))


# ----------------------------- ranking metrics ----------------------------- #

def precision_at_k(recommended, relevant_set, k):
    """
    Precision@K = (# of the top-K recommended items that are relevant) / K.

    "Of what we showed the user, how much of it was actually good?"
    `recommended` is a ranked list of item ids (best first); `relevant_set`
    is the set of item ids the user actually liked (held out).
    """
    if k <= 0:
        return 0.0
    top_k = recommended[:k]
    if len(top_k) == 0:
        return 0.0
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / k


def recall_at_k(recommended, relevant_set, k):
    """
    Recall@K = (# of the top-K recommended items that are relevant) / (total # relevant items).

    "Of everything the user actually liked, how much did we surface in the top K?"
    Undefined (returns 0.0) when the user has no relevant items at all.
    """
    if not relevant_set:
        return 0.0
    top_k = recommended[:k]
    hits = sum(1 for item in top_k if item in relevant_set)
    return hits / len(relevant_set)


def dcg_at_k(recommended, relevance, k):
    """
    Discounted Cumulative Gain@K = sum_{rank=1..K} (2^rel_i - 1) / log2(rank + 1).

    `relevance` maps item_id -> graded relevance score (here, the held-out
    star rating; 0 if not in the held-out set / never rated). Items placed
    higher in the ranking (`rank` closer to 1) count more, via the
    log2(rank+1) discount; the (2^rel - 1) gain term rewards highly-relevant
    items (a 5-star hit) disproportionately more than a marginal one
    (a 3-star hit), rather than linearly.
    """
    dcg = 0.0
    for rank, item in enumerate(recommended[:k], start=1):
        rel = relevance.get(item, 0)
        dcg += (2 ** rel - 1) / np.log2(rank + 1)
    return dcg


def ndcg_at_k(recommended, relevance, k):
    """
    Normalized DCG@K = DCG@K / IDCG@K, where IDCG@K is the DCG of the best
    *possible* ranking (all relevant items sorted by relevance, first).
    Normalizing bounds the score to [0, 1] and makes it comparable across
    users who have different numbers/levels of relevant items.
    """
    dcg = dcg_at_k(recommended, relevance, k)
    ideal_order = sorted(relevance.values(), reverse=True)[:k]
    idcg = sum((2 ** rel - 1) / np.log2(rank + 1) for rank, rel in enumerate(ideal_order, start=1))
    if idcg == 0:
        return 0.0
    return dcg / idcg


# ----------------------------- evaluation harness ----------------------------- #

def evaluate_rating_prediction(model, test_df, user_map, item_map):
    """Compute RMSE/MAE of model.predict(u_idx, i_idx) against held-out test ratings."""
    preds, trues = [], []
    for row in test_df.itertuples(index=False):
        if not (user_map.contains(row.user_id) and item_map.contains(row.item_id)):
            continue
        u = user_map.to_idx(row.user_id)
        i = item_map.to_idx(row.item_id)
        preds.append(model.predict(u, i))
        trues.append(row.rating)
    return {
        "rmse": rmse(trues, preds),
        "mae": mae(trues, preds),
        "n_predictions": len(preds),
    }


def evaluate_ranking(model, train_df, test_df, user_map, item_map, k=10, relevance_threshold=4.0):
    """
    For every test user, rank all items the user has NOT rated in train,
    take the top-K by model score, and compare against what they actually
    rated in the held-out test set.

    relevant_set (for Precision/Recall) = test items rated >= relevance_threshold.
    relevance dict (for NDCG) = graded, using the actual held-out star rating.

    Users with no held-out ratings are skipped entirely (nothing to
    evaluate against). Users with zero *relevant* (>=threshold) items are
    included in NDCG (their ideal ranking may still be nonzero if any
    lower graded item exists) but contribute 0 to precision/recall
    (correctly, since recall of "nothing relevant" is trivially 0 --
    consistent with recall_at_k's definition, not excluded, since a model
    that ranks their disliked items low should still not be double-penalized
    vs. one that doesn't).
    """
    train_by_user = train_df.groupby("user_id")["item_id"].apply(set)
    test_by_user = test_df.groupby("user_id")

    precisions, recalls, ndcgs = [], [], []
    n_users_evaluated = 0

    all_item_idx = np.arange(len(item_map))

    for user_id, group in test_by_user:
        if not user_map.contains(user_id):
            continue
        u = user_map.to_idx(user_id)

        rated_train_items = train_by_user.get(user_id, set())
        rated_train_idx = set(item_map.to_idx(i) for i in rated_train_items if item_map.contains(i))

        scores = model.predict_all_for_user(u).copy()
        if len(rated_train_idx) > 0:
            scores[list(rated_train_idx)] = -np.inf

        ranked_idx = np.argsort(-scores)[:k]
        recommended_items = [item_map.to_raw(idx) for idx in ranked_idx]

        test_items = group.set_index("item_id")["rating"].to_dict()
        relevant_set = {item for item, r in test_items.items() if r >= relevance_threshold}

        precisions.append(precision_at_k(recommended_items, relevant_set, k))
        recalls.append(recall_at_k(recommended_items, relevant_set, k))
        ndcgs.append(ndcg_at_k(recommended_items, test_items, k))
        n_users_evaluated += 1

    return {
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "n_users_evaluated": n_users_evaluated,
    }


def evaluate_model(model, train_df, test_df, user_map, item_map, k=10, relevance_threshold=4.0):
    """Run both rating-prediction and ranking evaluation and merge into one dict."""
    result = {}
    result.update(evaluate_rating_prediction(model, test_df, user_map, item_map))
    result.update(evaluate_ranking(model, train_df, test_df, user_map, item_map, k=k,
                                    relevance_threshold=relevance_threshold))
    return result
