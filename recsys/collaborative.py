"""
Collaborative filtering via matrix factorization (a simplified Funk-SVD),
trained with stochastic gradient descent (SGD).

--- The math ---

We want to explain the observed rating matrix R (users x items, mostly
missing) as:

    r_hat(u, i) = mu + b_u[u] + b_i[i] + P[u] . Q[i]

where:
    mu    = global average rating (a sensible prior for "unknown" pairs)
    b_u   = per-user bias (some users rate everything higher/lower on average)
    b_i   = per-item bias (some items are just liked more on average)
    P[u]  = a length-k latent factor vector for user u
    Q[i]  = a length-k latent factor vector for item i
    P[u] . Q[i] = dot product capturing user-item taste interaction
                  (e.g. a "sci-fi-ness" coordinate where high P[u][d] and
                  high Q[i][d] together push the predicted rating up)

We only observe a small fraction of the |U| x |I| matrix (the ratings a
user actually gave), so we minimize *regularized squared error* over just
the observed entries:

    L = sum_(u,i) in observed [ (r_ui - r_hat(u,i))^2 ]
        + reg * ( ||P[u]||^2 + ||Q[i]||^2 + b_u[u]^2 + b_i[i]^2 )

The L2 regularization term discourages the latent vectors from growing
large just to memorize noise in a handful of observations — it's what
keeps this "matrix completion" problem well-posed despite being wildly
underdetermined (way more free parameters than any single user or item
has observations).

We minimize L with SGD: for each observed (u, i, r) triple, compute the
prediction error e = r - r_hat(u, i), then nudge every involved parameter
a small step (learning rate `lr`) in the direction that reduces its
squared-error contribution, shrunk slightly by regularization:

    b_u[u] += lr * (e - reg * b_u[u])
    b_i[i] += lr * (e - reg * b_i[i])
    P[u]   += lr * (e * Q[i]_old - reg * P[u])
    Q[i]   += lr * (e * P[u]_old - reg * Q[i])

Repeating this over many epochs (full passes over the training ratings,
reshuffled each time) converges to a local minimum of L. This is exactly
the algorithm popularized by Simon Funk during the Netflix Prize.
"""
import numpy as np


class MatrixFactorization:
    def __init__(self, n_factors=20, lr=0.01, reg=0.02, n_epochs=25, seed=42):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.n_epochs = n_epochs
        self.seed = seed

        self.n_users = None
        self.n_items = None
        self.global_mean = 0.0
        self.b_u = None
        self.b_i = None
        self.P = None
        self.Q = None
        self.train_rmse_history = []

    def fit(self, u_idx, i_idx, ratings, n_users, n_items, verbose=False):
        """
        Train via SGD.

        u_idx, i_idx, ratings: parallel 1-D arrays of (user_idx, item_idx, rating)
            for every observed training entry.
        n_users, n_items: total counts, used to size the factor matrices
            (some users/items may have zero ratings in an edge case; they
            keep their randomly-initialized factors and fall back to bias
            terms / global mean at prediction time).
        """
        rng = np.random.default_rng(self.seed)
        self.n_users, self.n_items = n_users, n_items
        self.global_mean = float(np.mean(ratings))

        # Small random init: too-large initial factors make early gradients
        # explode; too-small (e.g. all zeros) means P.Q starts at exactly 0
        # for everyone, which is fine here since biases carry the signal
        # initially, but a small random spread breaks symmetry across factors.
        self.b_u = np.zeros(n_users, dtype=np.float64)
        self.b_i = np.zeros(n_items, dtype=np.float64)
        self.P = rng.normal(0, 0.1, size=(n_users, self.n_factors))
        self.Q = rng.normal(0, 0.1, size=(n_items, self.n_factors))

        n = len(ratings)
        order = np.arange(n)
        self.train_rmse_history = []

        for epoch in range(self.n_epochs):
            rng.shuffle(order)
            sq_err_sum = 0.0
            for k in order:
                u, i, r = u_idx[k], i_idx[k], ratings[k]
                pred = self.global_mean + self.b_u[u] + self.b_i[i] + self.P[u].dot(self.Q[i])
                err = r - pred
                sq_err_sum += err * err

                # Cache old P[u] before mutating it, since Q[i]'s update
                # needs the pre-update user vector (simultaneous gradient
                # step, not a sequential one).
                p_u_old = self.P[u].copy()
                q_i_old = self.Q[i].copy()

                self.b_u[u] += self.lr * (err - self.reg * self.b_u[u])
                self.b_i[i] += self.lr * (err - self.reg * self.b_i[i])
                self.P[u] += self.lr * (err * q_i_old - self.reg * p_u_old)
                self.Q[i] += self.lr * (err * p_u_old - self.reg * q_i_old)

            epoch_rmse = np.sqrt(sq_err_sum / n)
            self.train_rmse_history.append(epoch_rmse)
            if verbose:
                print(f"  epoch {epoch + 1}/{self.n_epochs}  train RMSE = {epoch_rmse:.4f}")

        return self

    def predict(self, u, i):
        """Predict a single rating, clipped to the valid [1, 5] rating scale."""
        if u >= self.n_users or i >= self.n_items or u < 0 or i < 0:
            return self.global_mean
        pred = self.global_mean + self.b_u[u] + self.b_i[i] + self.P[u].dot(self.Q[i])
        return float(np.clip(pred, 1.0, 5.0))

    def predict_all_for_user(self, u):
        """Vectorized prediction of every item's score for user u (unclipped, for ranking)."""
        if u >= self.n_users or u < 0:
            return np.full(self.n_items, self.global_mean)
        return self.global_mean + self.b_u[u] + self.b_i + self.Q.dot(self.P[u])

    def save(self, path):
        np.savez(
            path,
            n_factors=self.n_factors, lr=self.lr, reg=self.reg, n_epochs=self.n_epochs,
            n_users=self.n_users, n_items=self.n_items, global_mean=self.global_mean,
            b_u=self.b_u, b_i=self.b_i, P=self.P, Q=self.Q,
            train_rmse_history=np.array(self.train_rmse_history),
        )

    @classmethod
    def load(cls, path):
        d = np.load(path)
        model = cls(
            n_factors=int(d["n_factors"]), lr=float(d["lr"]), reg=float(d["reg"]),
            n_epochs=int(d["n_epochs"]),
        )
        model.n_users = int(d["n_users"])
        model.n_items = int(d["n_items"])
        model.global_mean = float(d["global_mean"])
        model.b_u = d["b_u"]
        model.b_i = d["b_i"]
        model.P = d["P"]
        model.Q = d["Q"]
        model.train_rmse_history = list(d["train_rmse_history"])
        return model
