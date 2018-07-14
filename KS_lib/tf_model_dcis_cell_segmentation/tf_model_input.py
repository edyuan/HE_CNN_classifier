import os
import tensorflow as tf
import collections
import numpy as np

from KS_lib.general import KScsv

#####################################################################################
def read_data(filename_queue, flags):

    image_content = tf.read_file(filename_queue[0])
    label_content = tf.read_file(filename_queue[1])
    weight_content = tf.read_file(filename_queue[2])

    image = tf.image.decode_png(image_content)
    label = tf.image.decode_png(label_content)
    weight = tf.image.decode_png(weight_content)

    image.set_shape(flags['size_input_patch'])
    label.set_shape(flags['size_output_patch'])
    weight.set_shape(flags['size_output_patch'])

    return image, label, weight

#####################################################################################
def process_image_and_label(image, label, weight, mean_image, variance_image, flags):
    # Randomly flip the image.
    r_flip = tf.random_uniform([3], 0, 1.0, dtype=tf.float32)
    # left right
    mirror = tf.less(tf.pack([1.0, r_flip[0], 1.0]), 0.5)
    image = tf.reverse(image, mirror)
    label = tf.reverse(label, mirror)
    weight = tf.reverse(weight, mirror)

    # up down
    mirror = tf.less(tf.pack([r_flip[1], 1.0, 1.0]), 0.5)
    image = tf.reverse(image, mirror)
    label = tf.reverse(label, mirror)
    weight = tf.reverse(weight, mirror)

    # transpose
    mirror = tf.less(tf.pack([r_flip[2], 1.0 - r_flip[2]]), 0.5)
    mirror = tf.cast(mirror, tf.int32)
    mirror = tf.pack([mirror[0], mirror[1], 2])

    image = tf.transpose(image, perm=mirror)
    label = tf.transpose(label, perm=mirror)
    weight = tf.transpose(weight, perm=mirror)
    image.set_shape(flags['size_input_patch'])
    label.set_shape(flags['size_output_patch'])
    weight.set_shape(flags['size_output_patch'])

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    # image = tf.image.random_brightness(image, max_delta=63)
    # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    # image = tf.image.random_hue(image, max_delta=0.2)
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    r, g, b = tf.split(2, 3, image)
    rr = tf.random_uniform([1], minval=np.log(0.25), maxval=np.log(4), dtype=tf.float32)
    rg = tf.random_uniform([1], minval=np.log(0.25), maxval=np.log(4), dtype=tf.float32)
    rb = tf.random_uniform([1], minval=np.log(0.25), maxval=np.log(4), dtype=tf.float32)

    r = 255.0 * tf.pow(r / 255.0, tf.exp(rr))
    g = 255.0 * tf.pow(g / 255.0, tf.exp(rg))
    b = 255.0 * tf.pow(b / 255.0, tf.exp(rb))
    image = tf.concat(2, [r, g, b])

    # random brightness field
    M = flags['size_input_patch'][0]*10
    N = flags['size_input_patch'][1]*10
    r0 = np.random.uniform(0, M, 10*10)
    r1 = np.random.uniform(0, N, 10*10)

    r0 = np.unique(np.rint(np.append(0, np.append(r0, M)))).astype(np.int)
    r1 = np.unique(np.rint(np.append(0, np.append(r1, N)))).astype(np.int)

    random_field_mat = np.zeros(shape=(M, N, 1), dtype=np.float32)
    for i in xrange(len(r0) - 1):
        for j in xrange(len(r1) - 1):
            random_field_mat[r0[i]:r0[i + 1], r1[j]:r1[j + 1], :] = np.random.uniform(0.5, 1.5, 1)

    big_random_field = tf.convert_to_tensor(random_field_mat, dtype = tf.float32)
    random_field = tf.random_crop(big_random_field, [flags['size_input_patch'][0],
                                                     flags['size_input_patch'][1],
                                                     1])

    random_field = tf.image.random_flip_up_down(random_field)
    random_field = tf.image.random_flip_left_right(random_field)

    hsv = tf.image.rgb_to_hsv(image)
    h, s, v = tf.split(2, 3, hsv)
    v = v*random_field
    hsv = tf.concat(2, [h, s, v])
    image = tf.image.hsv_to_rgb(hsv)

    image = tf.image.random_brightness(image, max_delta=63)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # image = tf.image.random_hue(image, max_delta=0.4)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)

    epsilon = 1e-6
    image = image - mean_image
    image = image / tf.sqrt(variance_image + epsilon)

    weight = weight/255.0

    return image, label, weight

#####################################################################################
def generate_batch(image, label, weight, min_queue_examples, batch_size, shuffle, flags):
    num_preprocess_threads = flags['num_preprocess_threads']
    if shuffle:
        images, labels, weights = tf.train.shuffle_batch(
            [image, label, weight],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, labels, weights = tf.train.batch(
            [image, label, weight],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    return images, labels, weights

#####################################################################################
def inputs(mean_image, variance_image, object_folder, mode, flags):
    if mode == 'train':
        log_file_path = os.path.join(object_folder, 'train', 'train_log.csv')
    else:
        log_file_path = os.path.join(object_folder, 'val', 'val_log.csv')

    log_list = KScsv.read_csv(log_file_path)

    image_dict = collections.defaultdict(list)
    label_dict = collections.defaultdict(list)
    weight_dict = collections.defaultdict(list)

    for row in log_list:
        image_dict['image'].append(row[0])
        label_dict['label'].append(row[1])
        weight_dict['weight'].append(row[2])

    min_queue_examples = int(len(image_dict['image']) * flags['min_fraction_of_examples_in_queue'])
    print('Filling queue with %d images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Create a queue that produces the filenames to read.
    combine_image_dict = list()
    combine_label_dict = list()
    combine_weight_dict = list()

    filename_queue = tf.train.slice_input_producer(
                     [image_dict['image'], label_dict['label'], weight_dict['weight']],
                     shuffle=True)
    image, label, weight  = read_data(filename_queue, flags)
    image = tf.cast(image, tf.float32)
    label = tf.cast(label, tf.float32)
    weight = tf.cast(weight, tf.float32)
    image, label, weight = process_image_and_label(image, label, weight, mean_image, variance_image, flags)

    # Generate a batch of images and labels by building up a queue of examples.
    image, label, weight = generate_batch(image, label, weight, min_queue_examples,
                                          int(flags['batch_size']), shuffle=False, flags = flags)
    combine_image_dict.append(image)
    combine_label_dict.append(label)
    combine_weight_dict.append(weight)

    out_image = tf.concat(0, combine_image_dict)
    out_label = tf.concat(0, combine_label_dict)
    out_weight = tf.concat(0, combine_weight_dict)

    return {'images':out_image, 'labels':out_label, 'weights':out_weight}