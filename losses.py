import theano
import theano.tensor as T


def log_loss(y, t, eps=1e-15):
    """
    cross entropy loss, summed over classes, mean over batches
    """
    y = T.clip(y, eps, 1 - eps)
    loss = -T.sum(t * T.log(y)) / y.shape[0].astype(theano.config.floatX)
    return loss


def accuracy_loss(y, t, eps=1e-15):
    y_ = T.cast(T.argmax(y, axis=1), 'int32')
    t_ = T.cast(T.argmax(t, axis=1), 'int32')

    # predictions = T.argmax(y, axis=1)
    return -T.mean(T.switch(T.eq(y_, t_), 1, 0))


def quad_kappa_loss(y, t, y_pow=1, eps=1e-15):
    num_scored_items = y.shape[0]
    num_ratings = 5
    tmp = T.tile(T.arange(0, num_ratings).reshape((num_ratings, 1)),
                 reps=(1, num_ratings)).astype(theano.config.floatX)
    weights = (tmp - tmp.T) ** 2 / (num_ratings - 1) ** 2

    y_ = y ** y_pow
    y_norm = y_ / (eps + y_.sum(axis=1).reshape((num_scored_items, 1)))

    hist_rater_a = y_norm.sum(axis=0)
    hist_rater_b = t.sum(axis=0)

    conf_mat = T.dot(y_norm.T, t)

    nom = T.sum(weights * conf_mat)
    denom = T.sum(weights * T.dot(hist_rater_a.reshape((num_ratings, 1)),
                                  hist_rater_b.reshape((1, num_ratings))) /
                  num_scored_items.astype(theano.config.floatX))

    return - (1 - nom / denom)


def quad_kappa_log_hybrid_loss(y, t, y_pow=1, log_scale=0.5, log_offset=0.50):
    log_loss_res = log_loss(y, t)
    kappa_loss_res = quad_kappa_loss(y, t, y_pow=y_pow)
    return kappa_loss_res + log_scale * (log_loss_res - log_offset)


def quad_kappa_log_hybrid_loss_clipped(
        y, t, y_pow=1, log_cutoff=0.9, log_scale=0.5):
    log_loss_res = log_loss(y, t)
    kappa_loss_res = quad_kappa_loss(y, t, y_pow=y_pow)
    return kappa_loss_res + log_scale * \
        T.clip(log_loss_res, log_cutoff, 10 ** 3)


def mse(y, t):
    return T.mean((y - t) ** 2)
