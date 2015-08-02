import cPickle as pickle
import re
import glob
import os
import sys
import time

import theano
import theano.tensor as T
import numpy as np
import pandas as p
import lasagne as nn

from utils import hms, architecture_string

# 0: dump .pkl file
# 1: dataset (all_train, test)
# 2: do submission (0 or 1)
# 3: tta_transfos (type transfos. default, small, etc.)
# 4: tta_times (number of TTA runs)
# 5: tta_ensemble_method (mean, log_mean, etc.)

dump_path = sys.argv[1]
model_data = pickle.load(open(dump_path, 'r'))

# Setting some vars for easier ref.
chunk_size = model_data['chunk_size'] * 2
batch_size = model_data['batch_size']

l_out = model_data['l_out']
l_ins = model_data['l_ins']

# Print some basic stuff about the model.
num_params = nn.layers.count_params(l_out)
print "\n\t\tNumber of parameters: %d" % num_params

model_arch = architecture_string(model_data['l_out'])

print model_arch

# Set up Theano stuff to compute output.
output = nn.layers.get_output(l_out, deterministic=True)
input_ndims = [len(nn.layers.get_output_shape(l_in))
               for l_in in l_ins]
xs_shared = [nn.utils.shared_empty(dim=ndim)
             for ndim in input_ndims]
idx = T.lscalar('idx')

givens = {}
for l_in, x_shared in zip(l_ins, xs_shared):
    givens[l_in.input_var] = x_shared[
        idx * batch_size:(idx + 1) * batch_size
    ]

compute_output = theano.function(
    [idx],
    output,
    givens=givens,
    on_unused_input='ignore'
)

if os.path.exists('/media/user/'):
    prefix_path = '/media/user/Extended_ext4/'
else:
    prefix_path = '/run/shm/'

print "Using prefix path: %s.\n" % prefix_path

# Getting wanted ids to get output for.
if sys.argv[2] == 'all_train':
    # TODO: have to do it with dataloader and images_train etc. for the
    # correct splits
    dataset = 'all_train'

    img_dir = prefix_path + 'train_ds2_crop/'
elif sys.argv[2] == 'test':
    # Test.
    dataset = 'test'
    img_dir = prefix_path + 'test_ds2_crop/'


# Get ids of imgs in directory.
def get_img_ids(img_dir):
    test_files = list(set(glob.glob(os.path.join(img_dir, "*.jpeg"))))
    test_ids = []

    prog = re.compile(r'(\d+)_(\w+)\.jpeg')
    for img_fn in test_files:
        test_id, test_side = prog.search(img_fn).groups()
        test_id = int(test_id)

        test_ids.append(test_id)

    return sorted(set(test_ids))

img_ids = get_img_ids(img_dir)

if len(img_ids) == 0:
    raise ValueError('No img ids!\n')

print "\n\nDoing prediction on %s set.\n" % dataset
print "\n\t%i test ids.\n" % len(img_ids)

# Create dataloader with the test ids.
from generators import DataLoader
data_loader = DataLoader()  # model_data['data_loader']
new_dataloader_params = model_data['data_loader_params']
new_dataloader_params.update({'images_test': img_ids})
data_loader.set_params(new_dataloader_params)

if 'paired_transfos' in model_data:
    paired_transfos = model_data['paired_transfos']
else:
    paired_transfos = False

print "\tChunk size: %i.\n" % chunk_size

num_chunks = int(np.ceil((2 * len(img_ids)) / float(chunk_size)))

if 'data_loader_no_transfos' in model_data:
    no_transfo_params = model_data['data_loader_no_transfos']
    default_transfo_params = model_data['data_loader_default_transfo_params']
else:
    no_transfo_params = data_loader.no_transfo_params
    default_transfo_params = data_loader.default_transfo_params

# The default gen with "no transfos".
test_gen = lambda: data_loader.create_fixed_gen(
    data_loader.images_test,
    chunk_size=chunk_size,
    prefix_train=img_dir,
    prefix_test=img_dir,
    transfo_params=no_transfo_params,
    paired_transfos=paired_transfos,
)


def do_pred(test_gen):
    outputs = []

    for e, (xs_chunk, chunk_shape, chunk_length) in enumerate(test_gen()):
        num_batches_chunk = int(np.ceil(chunk_length / float(batch_size)))

        print "Chunk %i/%i" % (e + 1, num_chunks)

        print "  load data onto GPU"
        for x_shared, x_chunk in zip(xs_shared, xs_chunk):
            x_shared.set_value(x_chunk)

        print "  compute output in batches"
        outputs_chunk = []
        for b in xrange(num_batches_chunk):
            out = compute_output(b)
            outputs_chunk.append(out)

        outputs_chunk = np.vstack(outputs_chunk)
        outputs_chunk = outputs_chunk[:chunk_length]
        outputs.append(outputs_chunk)

    return np.vstack(outputs)


if len(sys.argv) > 4:
    # Doing TTA.
    tta_transfos = sys.argv[4]
    tta_times = int(sys.argv[5])
    tta_ensemble_method = sys.argv[6]

    print "Doing %s TTA %i times ...\n\n" % (tta_transfos, tta_times)

    if tta_transfos == 'default':
        test_gen = lambda: data_loader.create_fixed_gen(
            data_loader.images_test,
            chunk_size=chunk_size,
            prefix_train=img_dir,
            prefix_test=img_dir,
            transfo_params=default_transfo_params,
            paired_transfos=paired_transfos,
        )
    elif tta_transfos == 'small':
        transfos = no_transfo_params.copy()
        transfos.update({
            'rotation': True,
            'rotation_range': (0, 360),
            'flip': True, 'flip_prob': 0.5,
            'zoom': True, 'zoom_prob': 0.3,
            'zoom_range': (0.00, 0.01),
            'brightness': True, 'brightness_range': (0.9, 1.1),
        })

        # Same for now, more coming later.
        test_gen = lambda: data_loader.create_fixed_gen(
            data_loader.images_test,
            chunk_size=chunk_size,
            prefix_train=img_dir,
            prefix_test=img_dir,
            transfo_params=transfos,
            paired_transfos=paired_transfos,
        )

    all_outputs = []
    for i in xrange(tta_times):
        start_time = time.time()

        print "\t\t\tTTA %i OF %i...\n\n" % (i + 1, tta_times)
        pred = do_pred(test_gen)
        all_outputs.append(pred)

        time_since_start = time.time() - start_time
        print "\nOne TTA iteration took %s.\n" % \
              hms(time_since_start)
        print "Estimated %s to go...\n\n" % \
              hms((tta_times - (i + 1)) * time_since_start)

    print "\n\nDone doing TTA predictions! Ensembling ...\n"
    if tta_ensemble_method == 'mean':
        outputs = np.mean(np.asarray(all_outputs), axis=0)
    elif tta_ensemble_method == 'log_mean':
        outputs = np.mean(np.log(1e-5 + np.asarray(all_outputs)), axis=0)

    subm_fn = 'subm/' + "%s--%s[%s][%s][%i][%s].csv" % \
                        (model_data['model_id'],
                         model_data['configuration'],
                         dataset,
                         tta_transfos,
                         tta_times,
                         tta_ensemble_method)
else:
    # Normal no transfo predict.
    outputs = do_pred(test_gen)
    subm_fn = 'subm/' + "%s--%s[%s].csv" % \
                        (model_data['model_id'],
                         model_data['configuration'],
                         dataset)


test_names = np.vstack([map(lambda x: str(x) + '_left', img_ids),
                        map(lambda x: str(x) + '_right', img_ids)]).T
test_names = test_names.reshape((-1, 1))

print "Saving...\n"
target_path = subm_fn[:-3].replace('subm', 'preds') + 'npy'
print test_names.shape
print outputs.shape

np.save(target_path, np.concatenate([test_names, outputs], axis=1))
print "  Outputs saved to %s.\n" % target_path


do_subm = int(sys.argv[3])
if do_subm:
    subm = p.DataFrame(test_names, columns=['image'])
    subm['level'] = np.argmax(outputs, axis=1)

    subm.to_csv(subm_fn, index=False)

    print "Submission saved to %s.\n" % subm_fn
