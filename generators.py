import Queue
import threading
import os

from PIL import ImageEnhance
from PIL import Image, ImageChops, ImageOps
import numpy as np

from utils import get_img_ids_from_iter

# A lot of this code is a bad fork from https://github.com/benanne/kaggle-ndsb.


def make_thumb(image, size=(80, 80), pad=False):
    # http://stackoverflow.com/questions/9103257/resize-image-
    # maintaining-aspect-ratio-and-making-portrait-and-landscape-images-e
    image.thumbnail(size, Image.BILINEAR)
    image_size = image.size

    if pad:
        thumb = image.crop((0, 0, size[0], size[1]))

        offset_x = max((size[0] - image_size[0]) / 2, 0)
        offset_y = max((size[1] - image_size[1]) / 2, 0)

        thumb = ImageChops.offset(thumb, offset_x, offset_y)
    else:
        thumb = ImageOps.fit(image, size, Image.BILINEAR, (0.5, 0.5))

    return thumb


def load_image_and_process(im, im_dst, dim_dst, output_shape=(80, 80),
                           prefix_path='data/train_ds5_crop/',
                           transfo_params=None,
                           rand_values=None):
    im = Image.open(prefix_path + im + '.jpeg', mode='r')

    sort_dim = list(np.sort(im.size))

    dim_dst[0] = sort_dim[1] / 700.0
    dim_dst[1] = sort_dim[0] / 700.0

    im_new = im

    # Dict to keep track of random values.
    chosen_values = {}

    if transfo_params.get('extra_width_crop', False):
        w, h = im_new.size

        if w / float(h) >= 1.3:
            cols_thres = np.where(
                np.max(
                    np.max(
                        np.asarray(im_new),
                        axis=2),
                    axis=0) > 35)[0]

            # Extra cond compared to orig crop.
            if len(cols_thres) > output_shape[0] // 2:
                min_x, max_x = cols_thres[0], cols_thres[-1]
            else:
                min_x, max_x = 0, -1

            im_new = im_new.crop((min_x, 0,
                                  max_x, h))

    if transfo_params.get('crop_height', False):
        w, h = im_new.size

        if w > 1 and 0.98 <= h / float(w) <= 1.02:
            # "Normal" without height crop, do height crop.
            im_new = im_new.crop((0, int(0.05 * h),
                                  w, int(0.95 * h)))

    if transfo_params.get('crop', False) and not \
            transfo_params.get('crop_after_rotation', False):
        if rand_values:
            do_crop = rand_values['do_crop']
        else:
            do_crop = transfo_params['crop_prob'] > np.random.rand()
        chosen_values['do_crop'] = do_crop

        if do_crop:
            out_w, out_h = im_new.size
            w_dev = int(transfo_params['crop_w'] * out_w)
            h_dev = int(transfo_params['crop_h'] * out_h)

            # If values are supplied.
            if rand_values:
                w0, w1 = rand_values['w0'], rand_values['w1']
                h0, h1 = rand_values['h0'], rand_values['h1']
            else:
                w0 = np.random.randint(0, w_dev + 1)
                w1 = np.random.randint(0, w_dev + 1)
                h0 = np.random.randint(0, h_dev + 1)
                h1 = np.random.randint(0, h_dev + 1)

            # Add params to dict.
            chosen_values['w0'] = w0
            chosen_values['w1'] = w1
            chosen_values['h0'] = h0
            chosen_values['h1'] = h1

            im_new = im_new.crop((0 + w0, 0 + h0,
                                  out_w - w1, out_h - h1))

    # if transfo_params.get('new_gen', False):
    #     im_new = im_new.crop(im_new.getbbox())
    # im_new = im_new.resize(map(lambda x: x*2, output_shape),
    # resample=Image.BICUBIC)

    if transfo_params.get('shear', False):
        # http://stackoverflow.com/questions/14177744/how-does-
        # perspective-transformation-work-in-pil
        if transfo_params['shear_prob'] > np.random.rand():
            # print 'shear'
            # TODO: No chosen values because shear not really used.
            shear_min, shear_max = transfo_params['shear_range']
            m = shear_min + np.random.rand() * (shear_max - shear_min)
            out_w, out_h = im_new.size
            xshift = abs(m) * out_w
            new_width = out_w + int(round(xshift))
            im_new = im_new.transform((new_width, out_h), Image.AFFINE,
                                      (1, m, -xshift if m > 0 else 0, 0, 1, 0),
                                      Image.BICUBIC)

    if transfo_params.get('rotation_before_resize', False):
        if rand_values:
            rotation_param = rand_values['rotation_param']
        else:
            rotation_param = np.random.randint(
                transfo_params['rotation_range'][0],
                transfo_params['rotation_range'][1])
        chosen_values['rotation_param'] = rotation_param

        im_new = im_new.rotate(rotation_param, resample=Image.BILINEAR,
                               expand=transfo_params.get('rotation_expand',
                                                         False))
        if transfo_params.get('rotation_expand',
                              False):
            im_new = im_new.crop(im_new.getbbox())

    if transfo_params.get('crop_after_rotation', False):
        if rand_values:
            do_crop = rand_values['do_crop']
        else:
            do_crop = transfo_params['crop_prob'] > np.random.rand()
        chosen_values['do_crop'] = do_crop

        if do_crop:
            out_w, out_h = im_new.size
            w_dev = int(transfo_params['crop_w'] * out_w)
            h_dev = int(transfo_params['crop_h'] * out_h)

            # If values are supplied.
            if rand_values:
                w0, w1 = rand_values['w0'], rand_values['w1']
                h0, h1 = rand_values['h0'], rand_values['h1']
            else:
                w0 = np.random.randint(0, w_dev + 1)
                w1 = np.random.randint(0, w_dev + 1)
                h0 = np.random.randint(0, h_dev + 1)
                h1 = np.random.randint(0, h_dev + 1)

            # Add params to dict.
            chosen_values['w0'] = w0
            chosen_values['w1'] = w1
            chosen_values['h0'] = h0
            chosen_values['h1'] = h1

            im_new = im_new.crop((0 + w0, 0 + h0,
                                  out_w - w1, out_h - h1))

    # im_new = im_new.thumbnail(output_shape, resample=Image.BILINEAR)
    if transfo_params.get('keep_aspect_ratio', False):
        im_new = make_thumb(im_new, size=output_shape,
                           pad=transfo_params['resize_pad'])
    else:
        im_new = im_new.resize(output_shape, resample=Image.BILINEAR)
    # im_new = im_new.resize(output_shape, resample=Image.BICUBIC)
    # im_new = im_new.resize(map(lambda x: int(x * 1.2), output_shape),
    # resample=Image.BICUBIC)
    # im_new = im_new.crop(im_new.getbbox())

    if transfo_params.get('rotation', False) \
            and not transfo_params.get('rotation_before_resize', False):
        if rand_values:
            rotation_param = rand_values['rotation_param']
        else:
            rotation_param = np.random.randint(
                transfo_params['rotation_range'][0],
                transfo_params['rotation_range'][1])
        chosen_values['rotation_param'] = rotation_param

        im_new = im_new.rotate(rotation_param, resample=Image.BILINEAR,
                               expand=transfo_params.get('rotation_expand',
                                                         False))
        if transfo_params.get('rotation_expand',
                              False):
            im_new = im_new.crop(im_new.getbbox())

    # im_new = im_new.resize(output_shape, resample=Image.BICUBIC)
    if transfo_params.get('contrast', False):
        contrast_min, contrast_max = transfo_params['contrast_range']
        if rand_values:
            contrast_param = rand_values['contrast_param']
        else:
            contrast_param = np.random.uniform(contrast_min, contrast_max)
        chosen_values['contrast_param'] = contrast_param

        im_new = ImageEnhance.Contrast(im_new).enhance(contrast_param)

    if transfo_params.get('brightness', False):
        brightness_min, brightness_max = transfo_params['brightness_range']
        if rand_values:
            brightness_param = rand_values['brightness_param']
        else:
            brightness_param = np.random.uniform(brightness_min,
                                                 brightness_max)
        chosen_values['brightness_param'] = brightness_param

        im_new = ImageEnhance.Brightness(im_new).enhance(brightness_param)

    if transfo_params.get('color', False):
        color_min, color_max = transfo_params['color_range']
        if rand_values:
            color_param = rand_values['color_param']
        else:
            color_param = np.random.uniform(color_min, color_max)
        chosen_values['color_param'] = color_param

        im_new = ImageEnhance.Color(im_new).enhance(color_param)

    if transfo_params.get('flip', False):
        if rand_values:
            do_flip = rand_values['do_flip']
        else:
            do_flip = transfo_params['flip_prob'] > np.random.rand()

        chosen_values['do_flip'] = do_flip

        if do_flip:
            im_new = im_new.transpose(Image.FLIP_LEFT_RIGHT)

    if output_shape[0] < 200 and False:
        # Otherwise too slow.
        # TODO: Disabled for now
        if 'rotation' in transfo_params and transfo_params['rotation']:
            if rand_values:
                rotation_param = rand_values['rotation_param2']
            else:
                rotation_param = np.random.randint(
                    transfo_params['rotation_range'][0],
                    transfo_params['rotation_range'][1])

            im_new = im_new.rotate(rotation_param, resample=Image.BILINEAR,
                                   expand=False)
            # im_new = im_new.crop(im_new.getbbox())
            chosen_values['rotation_param2'] = rotation_param

    if transfo_params.get('zoom', False):
        if rand_values:
            do_zoom = rand_values['do_zoom']
        else:
            do_zoom = transfo_params['zoom_prob'] > np.random.rand()
        chosen_values['do_zoom'] = do_zoom

        if do_zoom:
            zoom_min, zoom_max = transfo_params['zoom_range']
            out_w, out_h = im_new.size
            if rand_values:
                w_dev = rand_values['w_dev']
            else:
                w_dev = int(np.random.uniform(zoom_min, zoom_max) / 2 * out_w)
            chosen_values['w_dev'] = w_dev

            im_new = im_new.crop((0 + w_dev,
                                  0 + w_dev,
                                  out_w - w_dev,
                                  out_h - w_dev))

    # im_new = im_new.resize(output_shape, resample=Image.BILINEAR)
    if im_new.size != output_shape:
        im_new = im_new.resize(output_shape, resample=Image.BILINEAR)

    im_new = np.asarray(im_new).astype('float32') / 255
    im_dst[:] = np.rollaxis(im_new.astype('float32'), 2, 0)

    im.close()
    del im, im_new

    return chosen_values


def patches_gen_pairs(images, labels, p_x=80, p_y=80, num_channels=3,
                      chunk_size=1024,
                      num_chunks=100, rng=np.random,
                      prefix_path='data/train_ds5_crop/',
                      transfo_params=None,
                      paired_transfos=False):
    num_patients = len(images)

    for n in xrange(num_chunks):
        indices = rng.randint(0, num_patients, chunk_size // 2)

        chunk_x = np.zeros((chunk_size, num_channels, p_x, p_y),
                           dtype='float32')
        chunk_dim = np.zeros((chunk_size, 2), dtype='float32')
        # chunk_y = labels[indices].astype('float32')
        chunk_y = np.zeros((chunk_size,), dtype='int32')
        chunk_shape = np.zeros((chunk_size, num_channels), dtype='float32')

        for k, idx in enumerate(indices):
            # First eye.
            img = str(images[idx]) + '_left'
            chosen_values = load_image_and_process(
                img,
                im_dst=chunk_x[2 * k],
                dim_dst=chunk_dim[2 * k],
                output_shape=(p_x, p_y),
                prefix_path=prefix_path,
                transfo_params=transfo_params)
            chunk_shape[2 * k] = chunk_x[2 * k].shape
            chunk_y[2 * k] = labels[idx][0]

            # Second eye.
            img = str(images[idx]) + '_right'
            load_image_and_process(
                img,
                im_dst=chunk_x[2 * k + 1],
                dim_dst=chunk_dim[2 * k + 1],
                output_shape=(p_x, p_y),
                prefix_path=prefix_path,
                transfo_params=transfo_params,
                rand_values=chosen_values if paired_transfos else None)

            chunk_shape[2 * k + 1] = chunk_x[2 * k + 1].shape
            chunk_y[2 * k + 1] = labels[idx][1]

        yield chunk_x, chunk_dim, np.eye(5)[chunk_y].astype('float32'), \
            chunk_shape

# Get rid of relative imports.
main_dir = os.path.abspath(os.path.dirname(__file__))

import pandas as p
# Get all train ids to know if patient id is train or test.
train_labels = p.read_csv(os.path.join(main_dir, 'data/trainLabels.csv'))
all_train_patient_ids = set(get_img_ids_from_iter(train_labels.image))


def patches_gen_pairs_pseudolabel(images, labels, p_x=80, p_y=80,
                                  num_channels=3, chunk_size=1024,
                                  num_chunks=100, rng=np.random,
                                  prefix_train='data/train_ds5_crop/',
                                  prefix_test='data/test_ds5_crop/',
                                  transfo_params=None,
                                  paired_transfos=False):
    num_patients = len(images)

    for n in xrange(num_chunks):
        indices = rng.randint(0, num_patients, chunk_size // 2)

        chunk_x = np.zeros((chunk_size, num_channels, p_x, p_y),
                           dtype='float32')
        chunk_dim = np.zeros((chunk_size, 2), dtype='float32')
        chunk_y = np.zeros((chunk_size, 5), dtype='float32')

        chunk_shape = np.zeros((chunk_size, num_channels), dtype='float32')

        int_labels = len(labels.shape) < 3
        id_matrix = np.eye(5)

        for k, idx in enumerate(indices):
            # First check if img id is train or test.
            patient_id = images[idx]

            if patient_id in all_train_patient_ids:
                prefix_path = prefix_train
            else:
                prefix_path = prefix_test

            # First eye.
            img_id = str(patient_id) + '_left'
            chosen_values = load_image_and_process(
                img_id,
                im_dst=chunk_x[2 * k],
                dim_dst=chunk_dim[2 * k],
                output_shape=(p_x, p_y),
                prefix_path=prefix_path,
                transfo_params=transfo_params)
            chunk_shape[2 * k] = chunk_x[2 * k].shape

            if int_labels:
                chunk_y[2 * k] = id_matrix[int(labels[idx][0])]
            else:
                chunk_y[2 * k] = labels[idx][0]

            # Second eye.
            img_id = str(patient_id) + '_right'
            load_image_and_process(img_id, im_dst=chunk_x[2 * k + 1],
                                   dim_dst=chunk_dim[2 * k + 1],
                                   output_shape=(p_x, p_y),
                                   prefix_path=prefix_path,
                                   transfo_params=transfo_params,
                                   rand_values=chosen_values
                                   if paired_transfos else None)

            chunk_shape[2 * k + 1] = chunk_x[2 * k + 1].shape

            if int_labels:
                chunk_y[2 * k + 1] = id_matrix[int(labels[idx][1])]
            else:
                chunk_y[2 * k + 1] = labels[idx][1]

        yield chunk_x, chunk_dim, chunk_y, chunk_shape


def patches_gen_fixed_pairs(images, p_x=80, p_y=80, num_channels=3,
                            chunk_size=1024,
                            prefix_train='data/train_ds5_crop/',
                            prefix_test='data/test_ds5_crop/',
                            transfo_params=None,
                            paired_transfos=False):
    num_patients = len(images)
    num_chunks = int(np.ceil((2 * num_patients) / float(chunk_size)))

    idx = 0

    for n in xrange(num_chunks):
        chunk_x = np.zeros((chunk_size, num_channels, p_x, p_y),
                           dtype='float32')
        chunk_dim = np.zeros((chunk_size, 2), dtype='float32')
        chunk_shape = np.zeros((chunk_size, num_channels), dtype='float32')
        chunk_length = chunk_size

        for k in xrange(chunk_size // 2):
            if idx >= num_patients:
                chunk_length = 2 * k
                break

            patient_id = images[idx]

            if patient_id in all_train_patient_ids:
                prefix_path = prefix_train
            else:
                prefix_path = prefix_test

            img_id = str(patient_id) + '_left'

            chosen_values = load_image_and_process(
                img_id,
                im_dst=chunk_x[2 * k],
                dim_dst=chunk_dim[2 * k],
                output_shape=(p_x, p_y),
                prefix_path=prefix_path,
                transfo_params=transfo_params)

            chunk_shape[2 * k] = chunk_x[2 * k].shape

            img_id = str(images[idx]) + '_right'
            load_image_and_process(
                img_id,
                im_dst=chunk_x[2 * k + 1],
                dim_dst=chunk_dim[2 * k + 1],
                output_shape=(p_x, p_y),
                prefix_path=prefix_path,
                transfo_params=transfo_params,
                rand_values=chosen_values if paired_transfos else None)

            chunk_shape[2 * k + 1] = chunk_x[2 * k + 1].shape
            idx += 1

        yield chunk_x, chunk_dim, chunk_shape, chunk_length


# From https://github.com/benanne.
def buffered_gen_threaded(source_gen, buffer_size=2):
    """
    Generator that runs a slow source generator in a separate thread.
    Beware of the GIL!
    buffer_size: the maximal number of items to pre-generate
        (length of the buffer)
    """
    if buffer_size < 2:
        raise RuntimeError("Minimal buffer size is 2!")

    buffer = Queue.Queue(maxsize=buffer_size - 1)
    # the effective buffer size is one less, because the generation process
    # will generate one extra element and block until there is room in the
    # buffer.

    def _buffered_generation_thread(source_gen, buffer):
        for data in source_gen:
            buffer.put(data, block=True)
        buffer.put(None)  # sentinel: signal the end of the iterator

    thread = threading.Thread(target=_buffered_generation_thread,
                              args=(source_gen, buffer))
    thread.daemon = True
    thread.start()

    for data in iter(buffer.get, None):
        yield data


def buffered_gen_threaded_multiple(source_gens,
                                   buffer_size=3):
    """
    Generator that runs a slow source generator in a separate thread.
    Beware of the GIL!
    buffer_size: the maximal number of items to pre-generate
        (length of the buffer)
    """
    if buffer_size < 2:
        raise RuntimeError("Minimal buffer size is 2!")

    buffer = Queue.Queue(maxsize=buffer_size - 1)
    # the effective buffer size is one less, because the generation process
    # will generate one extra element and block until there is room in the
    # buffer.

    def _buffered_generation_thread(source_gen, buffer):
        for data in source_gen:
            buffer.put(data, block=True)
        buffer.put(None)  # sentinel: signal the end of the iterator

    for source_gen in source_gens:
        thread = threading.Thread(target=_buffered_generation_thread,
                                  args=(source_gen, buffer))
        thread.daemon = True
        thread.start()

    num_sentinels = 0

    while num_sentinels < len(source_gens):
        data = buffer.get()

        if data is not None:
            yield data
        else:
            num_sentinels += 1


class DataLoader(object):
    params = ['zmuv_mean', 'zmuv_std', 'p_x', 'p_y', 'num_channels', 'crop',
              'prefix_train', 'prefix_test',
              'default_transfo_params', 'no_transfo_params',
              'images_train_0', 'labels_train_0',
              'images_train_1', 'labels_train_1',
              'images_train_eval', 'labels_train_eval',
              'images_valid_eval', 'labels_valid_eval',
              'paired_transfos']

    paired_transfos = False

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def create_random_gen(self, images, labels, chunk_size=512,
                          num_chunks=100,
                          prefix_train='data/train_ds5_crop/',
                          prefix_test='data/test_ds5_crop/',
                          transfo_params=None,
                          buffer_size=3, num_generators=5,
                          paired_transfos=paired_transfos):
        if not transfo_params:
            raise ValueError("Need transfo_params for gen!")
            sys.exit(0)

        gens = []
        if num_generators > 1:
            for i in range(num_generators - 1):
                gen = patches_gen_pairs_pseudolabel(
                    images,
                    labels,
                    p_x=self.p_x,
                    p_y=self.p_y,
                    num_channels=self.num_channels,
                    chunk_size=chunk_size,
                    num_chunks=num_chunks //
                    num_generators,
                    prefix_train=prefix_train,
                    prefix_test=prefix_test,
                    transfo_params=transfo_params,
                    paired_transfos=paired_transfos)

                gens.append(gen)

        num_chunks_remaining = num_chunks - \
            (num_generators - 1) * (num_chunks // num_generators)

        gen = patches_gen_pairs_pseudolabel(images, labels,
                                            p_x=self.p_x, p_y=self.p_y,
                                            num_channels=self.num_channels,
                                            chunk_size=chunk_size,
                                            num_chunks=num_chunks_remaining,
                                            prefix_train=prefix_train,
                                            prefix_test=prefix_test,
                                            transfo_params=transfo_params,
                                            paired_transfos=paired_transfos)
        gens.append(gen)

        def random_gen(gen):
            for chunk_x, chunk_dim, chunk_y, chunk_shape in gen:
                yield [(chunk_x - self.zmuv_mean) /
                       (0.05 + self.zmuv_std),
                       chunk_dim], chunk_y, chunk_shape

        return buffered_gen_threaded_multiple(map(random_gen, gens),
                                              buffer_size=buffer_size)

    def create_fixed_gen(self, images, chunk_size=512,
                         prefix_train='data/train_ds5_crop/',
                         prefix_test='data/test_ds5_crop/',
                         buffer_size=2,
                         transfo_params=None,
                         paired_transfos=paired_transfos):

        if not transfo_params:
            raise ValueError("Need transfo_params for gen!")
            sys.exit(0)

        gen = patches_gen_fixed_pairs(images, p_x=self.p_x, p_y=self.p_y,
                                      num_channels=self.num_channels,
                                      chunk_size=chunk_size,
                                      prefix_train=prefix_train,
                                      prefix_test=prefix_test,
                                      transfo_params=transfo_params,
                                      paired_transfos=paired_transfos)

        def fixed_gen():
            for chunk_x, chunk_dim, chunk_shape, chunk_length in gen:
                yield [(chunk_x - self.zmuv_mean) /
                       (0.05 + self.zmuv_std),
                       chunk_dim], chunk_shape, chunk_length

        return buffered_gen_threaded(fixed_gen(), buffer_size=buffer_size)

    def estimate_params(self, transfo_params, eps=0.0,
                        pixel_based_norm=True):
        if self.num_channels > 3:
            paired = True
        else:
            paired = False

        gen = patches_gen_pairs_pseudolabel(self.images_train_0,
                                            self.labels_train_0,
                                            p_x=self.p_x, p_y=self.p_y,
                                            num_channels=self.num_channels,
                                            chunk_size=512,
                                            num_chunks=1,
                                            prefix_train=self.prefix_train,
                                            prefix_test=self.prefix_test,
                                            transfo_params=transfo_params,
                                            paired_transfos=paired)
        chunks_x, _, _, _ = gen.next()
        if pixel_based_norm:
            self.zmuv_mean = chunks_x.mean(axis=0, keepdims=True)
            self.zmuv_std = chunks_x.std(axis=0, keepdims=True) + eps
        else:
            self.zmuv_mean = chunks_x.mean(keepdims=True)
            self.zmuv_std = chunks_x.std(keepdims=True) + eps

        del chunks_x, gen

    def get_params(self):
        return {pname: getattr(self, pname, None)
                for pname in self.params}

    def set_params(self, p):
        self.__dict__.update(p)