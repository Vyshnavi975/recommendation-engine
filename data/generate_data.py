"""
Synthetic movie-ratings dataset generator.

Generates a realistic MovieLens-style dataset with:
  - users.csv    : user_id, favorite_genre, activity_level
  - items.csv    : item_id, title, genres (pipe-separated), tags (pipe-separated), popularity
  - ratings.csv  : user_id, item_id, rating, timestamp

Design goals for "realism" (so that a hybrid recommender has something
genuine to learn from):
  1. Each user has 1-2 preferred genres. Ratings on items matching those
     genres are biased upward; non-matching genres are biased downward.
  2. Items have an intrinsic "quality" latent factor, so some items are
     just broadly better/worse (independent of genre match).
  3. Item popularity follows a long-tail (power-law-ish) distribution, so
     a handful of items get rated very often and most get rated rarely,
     mirroring real catalog dynamics.
  4. User activity level varies (some users rate a lot, some rate little),
     also long-tailed.
  5. Gaussian noise is added on top so ratings aren't a deterministic
     function of genre match (otherwise the problem would be trivially
     easy and uninformative as an evaluation).

Ratings are on a 1-5 integer scale, clipped and rounded, like MovieLens.
"""
import os
import numpy as np
import pandas as pd

RNG_SEED = 42

GENRES = [
    "Action", "Comedy", "Drama", "Romance", "Sci-Fi",
    "Horror", "Thriller", "Animation", "Documentary", "Fantasy",
]

TAG_POOL = [
    "classic", "underrated", "cult-favorite", "blockbuster", "indie",
    "award-winning", "family-friendly", "dark", "feel-good", "twist-ending",
    "based-on-book", "franchise", "slow-burn", "visually-stunning", "dialogue-driven",
]

TITLE_ADJECTIVES = [
    "Eternal", "Silent", "Hidden", "Last", "Broken", "Golden", "Distant",
    "Forgotten", "Crimson", "Fractured", "Midnight", "Endless", "Sacred",
    "Wandering", "Shattered", "Velvet", "Neon", "Frozen", "Burning", "Quiet",
]
TITLE_NOUNS = [
    "Horizon", "Kingdom", "Shadow", "Journey", "Legacy", "Signal", "Garden",
    "Empire", "Voyage", "Echo", "Storm", "Harbor", "Labyrinth", "Reckoning",
    "Symphony", "Frontier", "Mirage", "Descent", "Ascension", "Paradox",
]


def generate_users(n_users, rng):
    user_ids = np.arange(1, n_users + 1)
    # Each user gets 1-2 favorite genres, drawn non-uniformly (some genres
    # are more popular as "favorites" than others, like real audiences).
    genre_popularity = rng.dirichlet(np.ones(len(GENRES)) * 2.0)
    n_fav = rng.choice([1, 2], size=n_users, p=[0.6, 0.4])
    fav_genres = []
    for k in n_fav:
        chosen = rng.choice(GENRES, size=k, replace=False, p=genre_popularity)
        fav_genres.append("|".join(chosen))

    # Activity level: long-tailed (log-normal) -> controls how many ratings
    # a user contributes.
    activity = rng.lognormal(mean=0.0, sigma=0.8, size=n_users)
    activity = activity / activity.mean()  # normalize around 1.0

    return pd.DataFrame({
        "user_id": user_ids,
        "favorite_genres": fav_genres,
        "activity_level": np.round(activity, 3),
    })


def generate_items(n_items, rng):
    item_ids = np.arange(1, n_items + 1)
    titles = []
    used = set()
    for _ in range(n_items):
        while True:
            t = f"The {rng.choice(TITLE_ADJECTIVES)} {rng.choice(TITLE_NOUNS)}"
            if t not in used:
                used.add(t)
                titles.append(t)
                break

    n_genre_pop = rng.dirichlet(np.ones(len(GENRES)) * 1.5)
    genres_col = []
    tags_col = []
    for _ in range(n_items):
        k = rng.choice([1, 2, 3], p=[0.5, 0.35, 0.15])
        g = rng.choice(GENRES, size=k, replace=False, p=n_genre_pop)
        genres_col.append("|".join(g))
        n_tags = rng.integers(1, 4)
        tg = rng.choice(TAG_POOL, size=n_tags, replace=False)
        tags_col.append("|".join(tg))

    # Popularity: long-tailed (Zipf-like via lognormal) -> controls how
    # often an item gets sampled for a rating, and also nudges quality.
    popularity = rng.lognormal(mean=0.0, sigma=1.1, size=n_items)

    # Intrinsic quality latent factor (item "goodness" independent of genre
    # match) - correlated a bit with popularity like in real catalogs
    # (popular things tend to be, on average, a little better-liked) but
    # with plenty of independent noise.
    quality = 0.3 * (np.log1p(popularity) - np.log1p(popularity).mean()) + rng.normal(0, 0.6, n_items)

    return pd.DataFrame({
        "item_id": item_ids,
        "title": titles,
        "genres": genres_col,
        "tags": tags_col,
        "popularity_weight": np.round(popularity, 4),
        "quality": np.round(quality, 4),
    })


def generate_ratings(users_df, items_df, n_ratings, rng):
    n_users = len(users_df)
    n_items = len(items_df)

    user_p = users_df["activity_level"].values
    user_p = user_p / user_p.sum()

    item_p = items_df["popularity_weight"].values
    item_p = item_p / item_p.sum()

    item_genres = items_df["genres"].str.split("|").tolist()
    item_quality = items_df["quality"].values
    user_fav_genres = users_df["favorite_genres"].str.split("|").tolist()

    # Oversample (user, item) pairs, then de-duplicate, to approach the
    # target rating count while avoiding a user rating the same item twice.
    seen = set()
    records = []
    target = n_ratings
    max_attempts = target * 6
    attempts = 0
    base_time = 1_600_000_000  # arbitrary Unix timestamp anchor

    while len(records) < target and attempts < max_attempts:
        attempts += 1
        batch = min(5000, (target - len(records)) * 3)
        u_idx = rng.choice(n_users, size=batch, p=user_p)
        i_idx = rng.choice(n_items, size=batch, p=item_p)
        for u, i in zip(u_idx, i_idx):
            if len(records) >= target:
                break
            key = (u, i)
            if key in seen:
                continue
            seen.add(key)

            # Genre-match bias: +1.1 average rating boost if the item's
            # genre set overlaps the user's favorite genres.
            overlap = bool(set(item_genres[i]) & set(user_fav_genres[u]))
            genre_bias = 1.1 if overlap else -0.3

            base = 3.2 + genre_bias + item_quality[i] * 0.8
            noise = rng.normal(0, 0.7)
            raw = base + noise
            rating = int(np.clip(np.round(raw), 1, 5))

            ts = base_time + int(rng.integers(0, 60 * 60 * 24 * 365 * 3))
            records.append((u + 1, i + 1, rating, ts))

    ratings_df = pd.DataFrame(records, columns=["user_id", "item_id", "rating", "timestamp"])
    ratings_df = ratings_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)
    return ratings_df


def main(n_users=800, n_items=250, n_ratings=35000, out_dir=None, seed=RNG_SEED):
    rng = np.random.default_rng(seed)
    out_dir = out_dir or os.path.join(os.path.dirname(__file__))
    os.makedirs(out_dir, exist_ok=True)

    users_df = generate_users(n_users, rng)
    items_df = generate_items(n_items, rng)
    ratings_df = generate_ratings(users_df, items_df, n_ratings, rng)

    users_df.to_csv(os.path.join(out_dir, "users.csv"), index=False)
    items_df.drop(columns=["popularity_weight", "quality"]).to_csv(
        os.path.join(out_dir, "items.csv"), index=False
    )
    ratings_df.to_csv(os.path.join(out_dir, "ratings.csv"), index=False)

    print(f"Generated {len(users_df)} users, {len(items_df)} items, {len(ratings_df)} ratings")
    print(f"Rating distribution:\n{ratings_df['rating'].value_counts().sort_index()}")
    print(f"Files written to: {out_dir}")


if __name__ == "__main__":
    main()
