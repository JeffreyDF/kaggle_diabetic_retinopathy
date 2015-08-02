import numpy as np

from utils import one_hot


def accuracy(y, t):
    if t.ndim == 2:
        t = np.argmax(t, axis=1)
    if y.ndim == 2:
        y = np.argmax(y, axis=1)

    return np.mean(y == t)


def log_losses(y, t, eps=1e-15):
    if t.ndim == 1:
        t = one_hot(t)

    y = np.clip(y, eps, 1 - eps)
    losses = -np.sum(t * np.log(y), axis=1)
    return losses


def continuous_kappa(y, t, y_pow=1, eps=1e-15):
    if y.ndim == 1:
        y = one_hot(y, m=5)

    if t.ndim == 1:
        t = one_hot(t, m=5)

    # Weights.
    num_scored_items, num_ratings = y.shape
    ratings_mat = np.tile(np.arange(0, num_ratings)[:, None],
                          reps=(1, num_ratings))
    ratings_squared = (ratings_mat - ratings_mat.T) ** 2
    weights = ratings_squared / float(num_ratings - 1) ** 2

    if y_pow != 1:
        y_ = y ** y_pow
        y_norm = y_ / (eps + y_.sum(axis=1)[:, None])
        y = y_norm

    hist_rater_a = np.sum(y, axis=0)
    hist_rater_b = np.sum(t, axis=0)

    conf_mat = np.dot(y.T, t)

    nom = weights * conf_mat
    denom = (weights * np.dot(hist_rater_a[:, None],
                              hist_rater_b[None, :]) /
             num_scored_items)

    return 1 - nom.sum() / denom.sum(), conf_mat, \
        hist_rater_a, hist_rater_b, nom, denom
