"""
Command-line interface for the recommendation engine.

Usage:
    python -m recsys.cli train    [options]
    python -m recsys.cli evaluate [options]
    python -m recsys.cli recommend --user-id 42 --top-k 10 [options]

Run `python -m recsys.cli <command> --help` for the full option list.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

from recsys.data import (
    DEFAULT_DATA_DIR, IdMap, load_data, train_test_split_per_user,
)
from recsys.collaborative import MatrixFactorization
from recsys.content_based import ContentBasedRecommender
from recsys.hybrid import HybridRecommender
from recsys.evaluation import evaluate_model

DEFAULT_MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)


# --------------------------------------------------------------------------- #
# persistence helpers
# --------------------------------------------------------------------------- #

def _paths(model_dir):
    return {
        "cf": os.path.join(model_dir, "cf_model.npz"),
        "cb": os.path.join(model_dir, "cb_model.pkl"),
        "maps": os.path.join(model_dir, "id_maps.pkl"),
        "config": os.path.join(model_dir, "config.json"),
        "metrics": os.path.join(model_dir, "metrics.json"),
    }


def _save_artifacts(model_dir, cf_model, cb_model, user_map, item_map, config):
    os.makedirs(model_dir, exist_ok=True)
    p = _paths(model_dir)
    cf_model.save(p["cf"])
    with open(p["cb"], "wb") as f:
        pickle.dump(cb_model, f)
    with open(p["maps"], "wb") as f:
        pickle.dump({"user_map": user_map, "item_map": item_map}, f)
    with open(p["config"], "w") as f:
        json.dump(config, f, indent=2)


def _load_artifacts(model_dir):
    p = _paths(model_dir)
    for key in ("cf", "cb", "maps", "config"):
        if not os.path.exists(p[key]):
            raise FileNotFoundError(
                f"Missing trained artifact: {p[key]}. Run `python -m recsys.cli train` first."
            )
    cf_model = MatrixFactorization.load(p["cf"])
    with open(p["cb"], "rb") as f:
        cb_model = pickle.load(f)
    with open(p["maps"], "rb") as f:
        maps = pickle.load(f)
    with open(p["config"]) as f:
        config = json.load(f)
    return cf_model, cb_model, maps["user_map"], maps["item_map"], config


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #

def cmd_train(args):
    t0 = time.time()
    print(f"Loading data from {args.data_dir} ...")
    ratings, items, users = load_data(args.data_dir)
    print(f"  {len(ratings)} ratings, {ratings['user_id'].nunique()} users, "
          f"{ratings['item_id'].nunique()} items")

    print(f"Splitting train/test (per-user holdout, test_frac={args.test_frac}) ...")
    train_df, test_df = train_test_split_per_user(
        ratings, test_frac=args.test_frac, min_ratings=args.min_ratings, seed=args.seed
    )
    print(f"  train: {len(train_df)} ratings | test: {len(test_df)} ratings")

    os.makedirs(args.data_dir, exist_ok=True)
    train_path = os.path.join(args.data_dir, "ratings_train.csv")
    test_path = os.path.join(args.data_dir, "ratings_test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"  saved split to {train_path}, {test_path}")

    # Build ID maps from the FULL ratings file so every user/item that
    # appears anywhere (train or test) gets a stable dense index.
    user_map = IdMap(ratings["user_id"].values)
    item_map = IdMap(items["item_id"].values)

    print(f"\nTraining collaborative filtering (matrix factorization) "
          f"[factors={args.n_factors}, lr={args.lr}, reg={args.reg}, epochs={args.epochs}] ...")
    u_idx = user_map.to_idx_array(train_df["user_id"].values)
    i_idx = item_map.to_idx_array(train_df["item_id"].values)
    r = train_df["rating"].values.astype(np.float64)

    cf_model = MatrixFactorization(
        n_factors=args.n_factors, lr=args.lr, reg=args.reg, n_epochs=args.epochs, seed=args.seed,
    )
    cf_model.fit(u_idx, i_idx, r, n_users=len(user_map), n_items=len(item_map), verbose=True)

    print("\nTraining content-based recommender (TF-IDF + cosine similarity) ...")
    cb_model = ContentBasedRecommender()
    cb_model.fit(train_df, items, user_map, item_map)
    print(f"  vocabulary size: {len(cb_model.vectorizer.vocabulary_)} genre/tag terms")
    print(f"  similarity->rating calibration: rating_hat = {cb_model.calib_a:.3f} "
          f"+ {cb_model.calib_b:.3f} * cosine_sim")

    config = {
        "alpha": args.alpha,
        "n_factors": args.n_factors,
        "lr": args.lr,
        "reg": args.reg,
        "epochs": args.epochs,
        "test_frac": args.test_frac,
        "min_ratings": args.min_ratings,
        "seed": args.seed,
        "k": args.k,
        "relevance_threshold": args.relevance_threshold,
    }
    _save_artifacts(args.model_dir, cf_model, cb_model, user_map, item_map, config)
    print(f"\nSaved trained models to {args.model_dir}/")
    print(f"Done in {time.time() - t0:.1f}s")


def cmd_evaluate(args):
    cf_model, cb_model, user_map, item_map, config = _load_artifacts(args.model_dir)
    alpha = args.alpha if args.alpha is not None else config.get("alpha", 0.7)
    k = args.k if args.k is not None else config.get("k", 10)
    threshold = args.relevance_threshold if args.relevance_threshold is not None \
        else config.get("relevance_threshold", 4.0)

    train_path = os.path.join(args.data_dir, "ratings_train.csv")
    test_path = os.path.join(args.data_dir, "ratings_test.csv")
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        print("No saved train/test split found. Run `python -m recsys.cli train` first.", file=sys.stderr)
        sys.exit(1)

    import pandas as pd
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    hybrid_model = HybridRecommender(cf_model, cb_model, alpha=alpha)

    results = {}
    print(f"Evaluating on {len(test_df)} held-out test ratings "
          f"({test_df['user_id'].nunique()} users) | K={k}, relevance_threshold={threshold}, "
          f"hybrid alpha={alpha}\n")

    for name, model in [
        ("Collaborative Filtering (MF)", cf_model),
        ("Content-Based (TF-IDF)", cb_model),
        (f"Hybrid (alpha={alpha})", hybrid_model),
    ]:
        m = evaluate_model(model, train_df, test_df, user_map, item_map, k=k,
                            relevance_threshold=threshold)
        results[name] = m

    header = f"{'Method':<32}{'RMSE':>8}{'MAE':>8}{f'P@{k}':>10}{f'R@{k}':>10}{f'NDCG@{k}':>10}"
    print(header)
    print("-" * len(header))
    for name, m in results.items():
        print(f"{name:<32}{m['rmse']:>8.4f}{m['mae']:>8.4f}"
              f"{m[f'precision@{k}']:>10.4f}{m[f'recall@{k}']:>10.4f}{m[f'ndcg@{k}']:>10.4f}")
    print(f"\n(evaluated ranking metrics over {list(results.values())[0]['n_users_evaluated']} "
          f"test users with held-out ratings)")

    metrics_path = _paths(args.model_dir)["metrics"]
    with open(metrics_path, "w") as f:
        json.dump({"k": k, "relevance_threshold": threshold, "alpha": alpha, "results": results}, f, indent=2)
    print(f"\nSaved metrics to {metrics_path}")


def cmd_recommend(args):
    cf_model, cb_model, user_map, item_map, config = _load_artifacts(args.model_dir)
    alpha = args.alpha if args.alpha is not None else config.get("alpha", 0.7)

    if not user_map.contains(args.user_id):
        print(f"Unknown user_id: {args.user_id}", file=sys.stderr)
        sys.exit(1)
    u = user_map.to_idx(args.user_id)

    if args.method == "cf":
        model = cf_model
    elif args.method == "content":
        model = cb_model
    else:
        model = HybridRecommender(cf_model, cb_model, alpha=alpha)

    import pandas as pd
    _, items, _ = load_data(args.data_dir)
    ratings, _, _ = load_data(args.data_dir)
    already_rated = set(ratings.loc[ratings["user_id"] == args.user_id, "item_id"].values)

    scores = model.predict_all_for_user(u).copy()
    rated_idx = [item_map.to_idx(i) for i in already_rated if item_map.contains(i)]
    if rated_idx:
        scores[rated_idx] = -np.inf

    top_idx = np.argsort(-scores)[: args.top_k]
    items_by_id = items.set_index("item_id")

    print(f"Top-{args.top_k} recommendations for user {args.user_id} "
          f"(method={args.method}{', alpha=' + str(alpha) if args.method == 'hybrid' else ''}):\n")
    print(f"{'Rank':<6}{'Score':<8}{'Title':<30}{'Genres':<25}{'Tags'}")
    print("-" * 100)
    for rank, idx in enumerate(top_idx, start=1):
        item_id = item_map.to_raw(idx)
        row = items_by_id.loc[item_id]
        score = scores[idx]
        print(f"{rank:<6}{score:<8.3f}{row['title']:<30}{row['genres']:<25}{row['tags']}")


# --------------------------------------------------------------------------- #
# argument parsing
# --------------------------------------------------------------------------- #

def build_parser():
    parser = argparse.ArgumentParser(prog="python -m recsys.cli", description="Hybrid recommendation engine CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="directory with users/items/ratings CSVs")
    common.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="directory to save/load trained models")

    p_train = sub.add_parser("train", parents=[common], help="train CF + content-based models and save them")
    p_train.add_argument("--n-factors", type=int, default=30, help="latent factor dimensionality for MF")
    p_train.add_argument("--lr", type=float, default=0.01, help="SGD learning rate")
    p_train.add_argument("--reg", type=float, default=0.05, help="L2 regularization strength")
    p_train.add_argument("--epochs", type=int, default=30, help="number of SGD epochs")
    p_train.add_argument("--test-frac", type=float, default=0.2, help="fraction of each user's ratings held out")
    p_train.add_argument("--min-ratings", type=int, default=5, help="min ratings/user required to hold any out")
    p_train.add_argument("--alpha", type=float, default=0.7, help="hybrid blend weight on CF score (default)")
    p_train.add_argument("--k", type=int, default=10, help="default K for ranking metrics")
    p_train.add_argument("--relevance-threshold", type=float, default=4.0,
                          help="min held-out rating counted as 'relevant' for precision/recall")
    p_train.add_argument("--seed", type=int, default=42)
    p_train.set_defaults(func=cmd_train)

    p_eval = sub.add_parser("evaluate", parents=[common], help="evaluate CF, content-based, and hybrid models")
    p_eval.add_argument("--alpha", type=float, default=None, help="override hybrid blend weight")
    p_eval.add_argument("--k", type=int, default=None, help="override K for ranking metrics")
    p_eval.add_argument("--relevance-threshold", type=float, default=None, help="override relevance threshold")
    p_eval.set_defaults(func=cmd_evaluate)

    p_rec = sub.add_parser("recommend", parents=[common], help="print top-K recommendations for a user")
    p_rec.add_argument("--user-id", type=int, required=True)
    p_rec.add_argument("--top-k", type=int, default=10)
    p_rec.add_argument("--method", choices=["cf", "content", "hybrid"], default="hybrid")
    p_rec.add_argument("--alpha", type=float, default=None, help="override hybrid blend weight")
    p_rec.set_defaults(func=cmd_recommend)

    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
