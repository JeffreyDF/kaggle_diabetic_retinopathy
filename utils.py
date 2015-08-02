import re
import glob
import os
import sys

import skimage
import numpy as np
import theano.tensor as T
from sklearn.cross_validation import StratifiedShuffleSplit

import string
import lasagne as nn


def padtosquare(im):
    w, l = im.shape

    if w < l:
        pad_size = (l - w) / 2.0
        im_new = skimage.util.pad(im, pad_width=((int(np.floor(pad_size)),
                                                  int(np.ceil(pad_size))),
                                                 (0, 0)),
                                  mode='constant',
                                  constant_values=(1, 1))
    else:
        pad_size = (w - l) / 2.0
        im_new = skimage.util.pad(im, pad_width=((0, 0),
                                                 (int(np.floor(pad_size)),
                                                  int(np.ceil(pad_size)))),
                                  mode='constant',
                                  constant_values=(1, 1))

    return im_new


def one_hot(vec, m=None):
    if m is None:
        m = int(np.max(vec)) + 1

    return np.eye(m)[vec].astype('int32')


def hms(seconds):
    seconds = np.floor(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return "%02d:%02d:%02d" % (hours, minutes, seconds)


def rms(x, axis=None, epsilon=1e-12):
    return T.sqrt(T.mean(T.sqr(x), axis=axis) + epsilon)


# TODO clean this mess up
def split_data(train_labels, labels_split, valid_size=20,
               SEED=42, stratified=True, pairs=False):
    if valid_size >= 100:
        return None

    num_all = len(train_labels)

    np.random.seed(SEED)

    if stratified:
        if pairs:
            # TODO: Taking max level to stratify for now.
            label_pairs = labels_split.groupby('id')['level'].max()
            label_pairs.index = map(int, label_pairs.index)
            label_pairs = label_pairs.sort_index(ascending=True)

            sss = StratifiedShuffleSplit(label_pairs.values, n_iter=1,
                                         test_size=0.01 * valid_size,
                                         indices=None, random_state=SEED)
        else:
            sss = StratifiedShuffleSplit(train_labels.level, n_iter=1,
                                         test_size=0.01 * valid_size,
                                         indices=None, random_state=SEED)
        for ix_train, ix_test in sss:
            pass
            # TODO: has no next(), need to figure this out
    else:
        shuffled_index = np.random.permutation(np.arange(num_all))

        num_valid = num_all // (100 / valid_size)
        num_train = num_all - num_valid

        ix_train = shuffled_index[:num_train]
        ix_test = shuffled_index[num_train:]

    if pairs:
        id_train = np.sort(np.asarray(label_pairs.index[ix_train]))
        y_train_left = labels_split[
            labels_split.id.isin(id_train)].level.values[::2]
        y_train_right = labels_split[
            labels_split.id.isin(id_train)].level.values[1::2]
        y_train = np.vstack([y_train_left, y_train_right]).T

        # TODO are they sorted
        assert labels_split[
            labels_split.id.isin(id_train)].eye[::2].unique().shape[0] == 1
        assert labels_split[
            labels_split.id.isin(id_train)].eye[1::2].unique().shape[0] == 1

        id_valid = np.sort(np.asarray(label_pairs.index[ix_test]))
        y_valid_left = labels_split[
            labels_split.id.isin(id_valid)].level.values[::2]
        y_valid_right = labels_split[
            labels_split.id.isin(id_valid)].level.values[1::2]
        y_valid = np.vstack([y_valid_left, y_valid_right]).T

        # TODO are they sorted
        assert labels_split[
            labels_split.id.isin(id_valid)].eye[::2].unique().shape[0] == 1
        assert labels_split[
            labels_split.id.isin(id_valid)].eye[1::2].unique().shape[0] == 1
    else:
        id_train = train_labels.ix[ix_train].image.values
        y_train = train_labels.ix[ix_train].level.values
        id_valid = train_labels.ix[ix_test].image.values
        y_valid = train_labels.ix[ix_test].level.values

    return id_train, y_train, id_valid, y_valid


# TODO: very ugly stuff here, can probably be done a lot better
def oversample_set(id_train, y_train, coefs):
    train_1 = list(np.where(np.apply_along_axis(
        lambda x: 1 in x,
        1,
        y_train))[0])
    train_2 = list(np.where(np.apply_along_axis(
        lambda x: 2 in x,
        1,
        y_train))[0])
    train_3 = list(np.where(np.apply_along_axis(
        lambda x: 3 in x,
        1,
        y_train))[0])
    train_4 = list(np.where(np.apply_along_axis(
        lambda x: 4 in x,
        1,
        y_train))[0])

    id_train_oversample = list(id_train)
    id_train_oversample += list(id_train[coefs[1] * train_1])
    id_train_oversample += list(id_train[coefs[2] * train_2])
    id_train_oversample += list(id_train[coefs[3] * train_3])
    id_train_oversample += list(id_train[coefs[4] * train_4])

    labels_train_oversample = np.array(y_train)
    labels_train_oversample = np.vstack([labels_train_oversample,
                                         y_train[coefs[1] * train_1]])
    labels_train_oversample = np.vstack([labels_train_oversample,
                                         y_train[coefs[2] * train_2]])
    labels_train_oversample = np.vstack([labels_train_oversample,
                                         y_train[coefs[3] * train_3]])
    labels_train_oversample = np.vstack([labels_train_oversample,
                                         y_train[coefs[4] * train_4]])

    return id_train_oversample, labels_train_oversample


def get_img_ids_from_iter(ar):
    test_ids = []

    prog = re.compile(r'\b(\d+)_(\w+)')

    for img_fn in ar:
        try:
            test_id, test_side = prog.search(img_fn).groups()
        except AttributeError:
            print img_fn
            sys.exit(0)

        test_id = int(test_id)

        test_ids.append(test_id)

    return test_ids


def get_img_ids_from_dir(img_dir):
    test_fns = glob.glob(os.path.join(img_dir, "*.jpeg"))
    return get_img_ids_from_iter(test_fns)


def softmax(ar, temp=1):
    e = np.exp(ar / temp)
    return e / e.sum(axis=1)[:, None]


def architecture_string(layer):
    model_arch = ''

    for i, layer in enumerate(nn.layers.get_all_layers(layer)):
        name = string.ljust(layer.__class__.__name__, 28)
        model_arch += "  %2i  %s %s  " % (i, name,
                                          nn.layers.get_output_shape(layer))

        if hasattr(layer, 'filter_size'):
            model_arch += str(layer.filter_size[0])
            model_arch += ' //'
        elif hasattr(layer, 'pool_size'):
            if isinstance(layer.pool_size, int):
                model_arch += str(layer.pool_size)
            else:
                model_arch += str(layer.pool_size[0])
            model_arch += ' //'
        if hasattr(layer, 'p'):
            model_arch += ' [%.2f]' % layer.p

        if hasattr(layer, 'stride'):
            model_arch += str(layer.stride[0])
        if hasattr(layer, 'learning_rate_scale'):
            if layer.learning_rate_scale != 1.0:
                model_arch += ' [lr_scale=%.2f]' % layer.learning_rate_scale
        if hasattr(layer, 'params'):
            for param in layer.params:
                if 'trainable' not in layer.params[param]:
                    model_arch += ' [NT] '

        model_arch += '\n'

    return model_arch
