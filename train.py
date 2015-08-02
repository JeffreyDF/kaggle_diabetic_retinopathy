# coding: utf-8
import os
import time
import datetime
import cPickle as pickle
from time import gmtime, strftime
from itertools import izip, izip_longest

import numpy as np
import theano
import theano.tensor as T

import lasagne as nn  # cf1a23c21666fc0225a05d284134b255e3613335
from utils import hms, architecture_string
from metrics import log_losses, accuracy, continuous_kappa

from models import basic_model as model

# theano.config.exception_verbosity = 'high'

import sys

if len(sys.argv) > 1:
    do_profile = int(sys.argv[1])
    print_graph = int(sys.argv[2])
else:
    do_profile = 0
    print_graph = 0


if do_profile:
    theano.config.profile = True
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# Set some vars from model.
LEARNING_RATE_SCHEDULE = model.LEARNING_RATE_SCHEDULE

prefix_train = model.prefix_train if hasattr(model, 'prefix_train') else \
    '/run/shm/train_ds2_crop/'
prefix_test = model.prefix_test if hasattr(model, 'prefix_test') else \
    '/run/shm/test_ds2_crop/'

SEED = model.SEED if hasattr(model, 'SEED') else 11111

id_train, y_train = model.id_train, model.y_train
id_valid, y_valid = model.id_valid, model.y_valid
id_train_oversample = model.id_train_oversample,
labels_train_oversample = model.labels_train_oversample

sample_coefs = model.sample_coefs if hasattr(model, 'sample_coefs') \
    else [0, 7, 3, 22, 25]

l_out, l_ins = model.build_model()
# l_ins = model.l_ins

chunk_size = model.chunk_size
batch_size = model.batch_size
num_chunks_train = model.num_chunks_train  # 5000
validate_every = model.validate_every  # 50
if hasattr(model, 'output_every'):
    output_every = model.output_every
else:
    output_every = validate_every

save_every = model.save_every  # 100
lr_decay = model.lr_decay if hasattr(model, 'lr_decay') else None

if lr_decay:
    lr_init = model.lr_init
    lr_final = model.lr_final
else:
    lr_init = LEARNING_RATE_SCHEDULE[1]

np.set_printoptions(precision=3)
np.set_printoptions(suppress=True)

model_id = strftime("%Y_%m_%d_%H%M%S", gmtime())


dump_path = 'dumps/' + model_id + '_' + model.config_name + '.pkl'

model_arch = architecture_string(l_out)
print model_arch

num_params = nn.layers.count_params(l_out, trainable=True)
print "\n\t\tNumber of trainable parameters: %d\n" % num_params
print "\t\tModel id: %s\n" % model_id
print "\t\tModel name: %s\n" % model.config_name

input_ndims = [len(nn.layers.get_output_shape(l_in))
               for l_in in l_ins]
xs_shared = [nn.utils.shared_empty(dim=ndim)
             for ndim in input_ndims]
y_shared = nn.utils.shared_empty(dim=2)

idx = T.lscalar('idx')

obj = model.build_objective(l_out)
train_loss = obj.get_loss()

output = nn.layers.get_output(l_out, deterministic=True)

givens = {
    obj.target_var: y_shared[
        idx * batch_size:(idx + 1) * batch_size
    ],
}

for l_in, x_shared in zip(l_ins, xs_shared):
    givens[l_in.input_var] = x_shared[
        idx * batch_size:(idx + 1) * batch_size
    ]

all_params = nn.layers.get_all_params(l_out, trainable=True)

learning_rate = theano.shared(np.array(lr_init,
                                       dtype=theano.config.floatX))
if hasattr(model, 'momentum'):
    momentum = model.momentum
else:
    momentum = 0.9

momentum = theano.shared(np.array(momentum,
                                  dtype=theano.config.floatX))


all_grads = T.grad(train_loss, all_params)

grads_norms = T.sqrt([T.sum(tensor**2) for tensor in all_grads])

scaled_grads = nn.updates.total_norm_constraint(all_grads,
                                                max_norm=10,
                                                return_norm=False)

updates = nn.updates.nesterov_momentum(scaled_grads, all_params,
                                       learning_rate, momentum)


iter_train = theano.function([idx], train_loss, givens=givens,
                             updates=updates)
compute_output = theano.function([idx], output, givens=givens,
                                 on_unused_input="ignore")

# iter_train has no timings now
if print_graph:
    theano.printing.debugprint(iter_train)

num_chunks = int((2 * len(y_train)) / float(chunk_size)) + 1

print "\t\tNum chunks per whole trainset (oversampled): %i.\n" % num_chunks

images_train_eval = model.images_train_eval
labels_train_eval = model.labels_train_eval
images_valid_eval = model.images_valid_eval
labels_valid_eval = model.labels_valid_eval

images_train_0 = model.images_train_0
labels_train_0 = model.labels_train_0
images_train_1 = model.images_train_1
labels_train_1 = model.labels_train_1

from generators import DataLoader

default_transfo_params = model.default_transfo_params
no_transfo_params = model.no_transfo_params
if hasattr(model, 'paired_transfos'):
    paired_transfos = model.paired_transfos
else:
    paired_transfos = False

data_loader = DataLoader(
    images_train_0=images_train_0,
    labels_train_0=labels_train_0,
    images_train_1=images_train_1,
    labels_train_1=labels_train_1,
    images_train_eval=images_train_eval,
    labels_train_eval=labels_train_eval,
    images_valid_eval=images_valid_eval,
    labels_valid_eval=labels_valid_eval,
    p_x=model.output_size,
    p_y=model.output_size,
    num_channels=model.num_channels,
    prefix_train=prefix_train,
    prefix_test=prefix_test,
    default_transfo_params=default_transfo_params,
    no_transfo_params=no_transfo_params,
)

print "Estimating parameters ..."
start = time.time()

if hasattr(model, 'pixel_based_norm'):
    pixel_based_norm = model.pixel_based_norm
else:
    pixel_based_norm = True

data_loader.estimate_params(transfo_params=no_transfo_params,
                            pixel_based_norm=pixel_based_norm)
end = time.time()
print "Done. (%.2f s)\n" % (end - start)

buffer_size = model.buffer_size
num_generators = model.num_generators


def oversample_sched(switch_chunk):
    first_gen = lambda: data_loader.create_random_gen(
        images=data_loader.images_train_0,
        labels=data_loader.labels_train_0,
        chunk_size=chunk_size,
        num_chunks=switch_chunk,
        prefix_train=data_loader.prefix_train,
        prefix_test=data_loader.prefix_test,
        transfo_params=default_transfo_params,
        paired_transfos=paired_transfos,
        buffer_size=buffer_size, num_generators=num_generators,
    )

    for elem in first_gen():
        yield elem

    print "\n\t\t\t SWITCHING GENERATORS ... \n"

    second_gen = lambda: data_loader.create_random_gen(
        images=data_loader.images_train_1,
        labels=data_loader.labels_train_1,
        chunk_size=chunk_size,
        num_chunks=num_chunks_train - switch_chunk,
        prefix_train=data_loader.prefix_train,
        prefix_test=data_loader.prefix_test,
        transfo_params=default_transfo_params,
        paired_transfos=paired_transfos,
        buffer_size=buffer_size, num_generators=num_generators,
    )

    for elem in second_gen():
        yield elem


switch_chunk = model.switch_chunk if hasattr(model, 'switch_chunk') \
    else num_chunks_train // 2

create_train_gen = lambda: oversample_sched(switch_chunk=switch_chunk)

create_eval_valid_gen = lambda: data_loader.create_fixed_gen(
    images=data_loader.images_valid_eval,
    prefix_train=data_loader.prefix_train,
    prefix_test=data_loader.prefix_test,
    transfo_params=no_transfo_params,
    paired_transfos=paired_transfos,
    chunk_size=chunk_size * 2,
    buffer_size=2,
)

# TODO: only 20% of train data, double chunk size, edited labels in train
# loop below
create_eval_train_gen = lambda: data_loader.create_fixed_gen(
    images=data_loader.images_train_eval[::5],
    prefix_train=data_loader.prefix_train,
    prefix_test=data_loader.prefix_test,
    transfo_params=no_transfo_params,
    paired_transfos=paired_transfos,
    chunk_size=chunk_size * 2,
    buffer_size=2,
)

num_batches_chunk = chunk_size // batch_size

print "Num batches per chunk: %i." % num_batches_chunk
print "Chunk size: %i.\n" % chunk_size
print "Num train chunks: %i.\n" % num_chunks_train
print "Batch size: %i.\n" % batch_size

chunks_train_ids = range(num_chunks_train)
losses_train = [np.inf]
losses_eval_valid = [np.inf]
losses_eval_train = [np.inf]

acc_eval_valid = [0]
acc_eval_train = [0]

metric_eval_valid = [0]
metric_eval_train = [0]

metric_extra_eval_valid = []
metric_extra_eval_train = []

metric_cont_eval_valid = [0]
metric_cont_eval_train = [0]

metric_cont_extra_eval_valid = []
metric_cont_extra_eval_train = []

learning_rate.set_value(LEARNING_RATE_SCHEDULE[1])

start_time = time.time()
prev_time = start_time

all_layers = nn.layers.get_all_layers(l_out)
diag_out = theano.function(
    [idx],
    nn.layers.get_output(all_layers, deterministic=True) + [grads_norms],
    givens=givens,
    on_unused_input="ignore"
)

for e, (xs_chunk, y_chunk, chunk_shape) in izip(chunks_train_ids,
                                                create_train_gen()):
    print "  Time waited: %.2f s.\n" % (time.time() - prev_time)

    print "Chunk %d/%d (next validation is in %d chunks)" % (
        e + 1, num_chunks_train,
        validate_every - ((e + 1) % validate_every)
    )

    # Linear lr decay every 50 chunks. TODO: cleanup
    if lr_decay == 'linear' and e % 50 == 0:
        lr = np.float32(
            lr_init - (lr_init - lr_final) *
            e / float(num_chunks_train)
        )
        print "  setting learning rate to %.7f (linear)\n" % lr
        learning_rate.set_value(lr)
    elif lr_decay == 'exp' and e % 50 == 0:
        lr = np.float32(
            lr_init * (lr_final /
                       float(lr_init)) ** (e / float(num_chunks_train))
        )
        print "  setting learning rate to %.7f (exponential)\n" % lr
        learning_rate.set_value(lr)
    else:
        if e + 1 in LEARNING_RATE_SCHEDULE:
            lr = np.float32(LEARNING_RATE_SCHEDULE[e + 1])
            print "  setting learning rate to %.7f\n" % lr
            learning_rate.set_value(lr)
            print "  learning rate schedule is:\n"
            print LEARNING_RATE_SCHEDULE
            print

    print "  load training data onto GPU"
    for x_shared, x_chunk in zip(xs_shared, xs_chunk):
        x_shared.set_value(x_chunk)

    y_shared.set_value(y_chunk)

    print "  batch SGD"
    losses = []
    for b in xrange(num_batches_chunk):
        loss = iter_train(b)

        if np.isnan(loss):
            raise RuntimeError("NaN DETECTED.")
        losses.append(loss)

    mean_train_loss = np.mean(losses)
    print "  mean training loss:\t\t%.6f" % mean_train_loss
    losses_train.append(mean_train_loss)

    if ((e + 1) % output_every) == 0:
        print '\n%2s  %7s  %7s  %7s  %7s - [%7s]' % (
            'n', 'MIN', 'MEAN', 'MAX', 'STD', 'NORM',
        )

        diag_result = diag_out(0)

        layers_out = diag_result[:-1]
        norms = diag_result[-1]

        # This is unaligned (because of the norms and layers can have
        # multiple params etc.) and messy but I have no time ...
        for i, (layer, norm) in enumerate(izip_longest(layers_out, norms,
                                                       fillvalue=0)):
            print '%2i  %7.2f  %7.2f  %7.2f  %7.2f - [%7.2f]' % (
                i, np.min(layer), np.mean(layer), np.max(layer),
                np.std(layer), norm
            )

        del diag_result, layers_out, norms

    if ((e + 1) % validate_every) == 0 or ((e + 1) == num_chunks_train):
        print
        print "Validating"
        subsets = ["train", "valid"]
        gens = [create_eval_train_gen, create_eval_valid_gen]

        # TODO: only 20% of training data
        label_sets = [np.eye(5)[data_loader.labels_train_eval[::5].flatten()],
                      np.eye(5)[data_loader.labels_valid_eval.flatten()]]
        losses_eval = [losses_eval_train,
                       losses_eval_valid]
        acc_eval = [acc_eval_train,
                    acc_eval_valid]
        metrics_eval = [metric_eval_train,
                        metric_eval_valid]
        metrics_extra = [metric_extra_eval_train,
                         metric_extra_eval_valid]
        metrics_cont_eval = [metric_cont_eval_train,
                             metric_cont_eval_valid]
        metrics_cont_extra = [metric_cont_extra_eval_train,
                              metric_cont_extra_eval_valid]

        for subset, create_gen, labels, losses, accs, metrics, metrics_extra,\
                metrics_cont, metrics_cont_extra in zip(
                    subsets, gens, label_sets, losses_eval, acc_eval,
                    metrics_eval, metrics_extra, metrics_cont_eval,
                    metrics_cont_extra):
            print "  %s set" % subset
            outputs = []
            for xs_chunk_eval, chunk_shape_eval, \
                    chunk_length_eval in create_gen():
                num_batches_chunk_eval = int(np.ceil(chunk_length_eval /
                                                     float(batch_size)))

                for x_shared, x_chunk_eval in zip(xs_shared, xs_chunk_eval):
                    x_shared.set_value(x_chunk_eval)

                outputs_chunk = []
                for b in xrange(num_batches_chunk_eval):
                    out = compute_output(b)
                    outputs_chunk.append(out)

                outputs_chunk = np.vstack(outputs_chunk)
                outputs_chunk = outputs_chunk[:chunk_length_eval]
                outputs.append(outputs_chunk)

            outputs = np.vstack(outputs)

            outputs_labels = np.argmax(outputs, axis=1)
            loss = np.mean(log_losses(outputs, labels))
            acc = accuracy(outputs_labels, labels)

            kappa_eval = continuous_kappa(
                outputs_labels,
                labels,
            )

            metric, conf_mat, \
                hist_rater_a, hist_rater_b, \
                nom, denom = kappa_eval

            try:
                kappa_cont_eval = continuous_kappa(outputs, labels,
                                                   y_pow=model.y_pow)
            except:
                kappa_cont_eval = [1, 1, 1, 1, 1, 1]

            metric_cont, conf_mat_cont, \
                hist_cont_rater_a, hist_cont_rater_b, \
                cont_nom, cont_denom = kappa_cont_eval

            print "    loss:\t%.6f \t%.6f \t BEST: %.6f" % (
                loss,
                loss - losses[-1],
                np.min(losses),
            )
            print "    acc:\t%.2f%% \t\t%.2f%% \t\t BEST: %.2f%%" % (
                acc * 100,
                (acc - accs[-1]) * 100,
                np.max(accs) * 100,
            )
            print "    quad kappa:\t%.3f \t\t%.3f \t\t BEST: %.3f" % (
                metric,
                metric - metrics[-1],
                np.max(metrics),
            )
            print "    confusion matrix: \n\n%s\n" % (
                conf_mat,
            )
            print "    normalised nom and denom: \n\n" \
                  " %s   (sum %.2f) \n\n" \
                  " %s   (sum %.2f) \n\n" % (
                      nom / nom.sum(),
                      nom.sum(),
                      denom / denom.sum(),
                      denom.sum(),
                  )
            print "    cont kappa:\t%.3f \t\t%.3f \t\t BEST: %.3f" % (
                metric_cont,
                metric_cont - metrics_cont[-1],
                np.max(metrics_cont),
            )
            print "    continuous confusion matrix: \n\n%s\n" % (
                conf_mat_cont,
            )
            print "    normalised continuous nom and denom: \n\n" \
                  " %s   (sum %.2f) \n\n" \
                  " %s   (sum %.2f) \n\n" % (
                      cont_nom / cont_nom.sum(),
                      cont_nom.sum(),
                      cont_denom / cont_denom.sum(),
                      cont_denom.sum(),
                  )

            losses.append(loss)
            accs.append(acc)
            metrics.append(metric)
            metrics_extra.append([conf_mat,
                                  hist_rater_a,
                                  hist_rater_b,
                                  nom,
                                  denom])
            metrics_cont.append(metric_cont)
            metrics_cont_extra.append([conf_mat_cont,
                                       hist_cont_rater_a,
                                       hist_cont_rater_b,
                                       cont_nom,
                                       cont_denom])
            del outputs

    now = time.time()
    time_since_start = now - start_time
    time_since_prev = now - prev_time
    prev_time = now
    est_time_left = time_since_start * \
        ((num_chunks_train - (e + 1)) /
         float(e + 1 - chunks_train_ids[0]))
    eta = datetime.datetime.now() + \
        datetime.timedelta(seconds=est_time_left)
    eta_str = eta.strftime("%c")

    print "  %s since start (%.2f s)" % (
        hms(time_since_start),
        time_since_prev
    )
    print "  estimated %s to go (ETA: %s)\n" % (
        hms(est_time_left),
        eta_str
    )

    # Save after every validate.
    if (((e + 1) % save_every) == 0 or
        ((e + 1) % validate_every) == 0 or
            ((e + 1) == num_chunks_train)):
        print "\nSaving model ..."

        with open(dump_path, 'w') as f:
            pickle.dump({
                'configuration': model.config_name,
                'model_id': model_id,
                'chunks_since_start': e,
                'time_since_start': time_since_start,
                'batch_size': batch_size,
                'chunk_size': chunk_size,
                'obj_loss': model.obj_loss,
                'l_ins': l_ins,
                'l_out': l_out,
                'lr_schedule': LEARNING_RATE_SCHEDULE,
                'lr_decay': lr_decay,
                'output_size': model.output_size,
                'data_loader_params': data_loader.get_params(),
                'sample_coefs': sample_coefs,
                'prefix_train': prefix_train,
                'switch_chunk': switch_chunk,
                'pl_enabled': model.pl_enabled,
                'pl_train_fn': model.pl_train_fn,
                'pl_test_fn': model.pl_test_fn,
                'pl_softmax_temp': model.pl_softmax_temp,
                'pl_train_coef': model.pl_train_coef,
                'pl_log': model.pl_log,
                'leakiness': model.leakiness,
                'SEED': SEED,
                'losses_train': losses_train,
                'losses_eval_valid': losses_eval_valid,
                'losses_eval_train': losses_eval_train,
                'acc_eval_valid': acc_eval_valid,
                'acc_eval_train': acc_eval_train,
                'metric_eval_valid': metric_eval_valid,
                'metric_eval_train': metric_eval_train,
                'metric_extra_eval_valid': metric_extra_eval_valid,
                'metric_extra_eval_train': metric_extra_eval_train,
                'metric_cont_eval_valid': metric_cont_eval_valid,
                'metric_cont_eval_train': metric_cont_eval_train,
                'metric_cont_extra_eval_valid': metric_cont_extra_eval_valid,
                'metric_cont_extra_eval_train': metric_cont_extra_eval_train,
                'y_pow': model.y_pow,
                'log_cutoff': model.log_cutoff,
                'lambda_reg': model.lambda_reg,
                'pixel_based_norm': pixel_based_norm,
                'paired_transfos': paired_transfos,
            }, f, pickle.HIGHEST_PROTOCOL)

        print "  saved to %s\n" % dump_path


print "\n\nTHE END."
