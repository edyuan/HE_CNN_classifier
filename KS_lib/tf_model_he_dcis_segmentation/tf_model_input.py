import os
import tensorflow as tf
import collections
import numpy as np

from KS_lib.general import KScsv

#####################################################################################
def read_data(filename_queue,flags):

    image_content = tf.read_file(filename_queue[0])
    label_content = filename_queue[1]

    image = tf.image.decode_png(image_content)
    label = label_content

    image.set_shape(flags['size_input_patch'])

    return image, label

#####################################################################################
def process_image_and_label(image, label, mean_image, variance_image, flags):
    # Randomly flip the image.
    r_flip = tf.random_uniform([3], 0, 1.0, dtype=tf.float32)

    # left right
    mirror = tf.less(tf.pack([1.0, r_flip[0], 1.0]), 0.5)
    image = tf.reverse(image, mirror)

    # up down
    mirror = tf.less(tf.pack([r_flip[1], 1.0, 1.0]), 0.5)
    image = tf.reverse(image, mirror)

    # transpose
    mirror = tf.less(tf.pack([r_flip[2], 1.0 - r_flip[2]]), 0.5)
    mirror = tf.cast(mirror, tf.int32)
    mirror = tf.pack([mirror[0], mirror[1], 2])

    image = tf.transpose(image, perm=mirror)
    image.set_shape(flags['size_input_patch'])

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    image = tf.image.random_brightness(image, max_delta=63)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # image = tf.image.random_hue(image, max_delta=0.4)

    r, g, b = tf.split(2, 3, image)
    rr = tf.random_uniform([1], minval=np.log(0.25), maxval=np.log(4), dtype=tf.float32)
    rb = tf.random_uniform([1], minval=np.log(0.25), maxval=np.log(4), dtype=tf.float32)
    r = 255.0 * tf.pow(r / 255.0, tf.exp(rr))
    b = 255.0 * tf.pow(b / 255.0, tf.exp(rb))
    image = tf.concat(2, [r, g, b])

    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)

    epsilon = 1e-6
    image = image - mean_image
    image = image / tf.sqrt(variance_image + epsilon)

    return image, label


#####################################################################################
def generate_batch(image, label, min_queue_examples, batch_size, shuffle, flags):
    num_preprocess_threads = flags['num_preprocess_threads']
    if shuffle:
        images, labels = tf.train.shuffle_batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        images, labels = tf.train.batch(
            [image, label],
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    return images, labels


#####################################################################################
def inputs(mean_image, variance_image, object_folder, mode, flags):
    if mode == 'train':
        log_file_path = os.path.join(object_folder, 'train', 'train_log.csv')
    else:
        log_file_path = os.path.join(object_folder, 'val', 'val_log.csv')

    log_list = KScsv.read_csv(log_file_path)

    image_dict = collections.defaultdict(list)
    label_dict = collections.defaultdict(list)

    for row in log_list:
        for i_class in range(flags['n_classes']):
            if int(row[2]) == i_class:
                image_dict[i_class].append(row[0])
                label_dict[i_class].append(int(row[2]))
    min_queue_examples = int(
        np.sum([len(image_dict[k]) for k in image_dict.keys()]) * flags['min_fraction_of_examples_in_queue'])

    print('Filling queue with %d images before starting to train. '
          'This will take a few minutes.' % min_queue_examples)

    # Create a queue that produces the filenames to read.
    combine_image_dict = list()
    combine_label_dict = list()
    for i_class in range(flags['n_classes']):
        filename_queue = tf.train.slice_input_producer([image_dict[i_class], label_dict[i_class]], shuffle=True)
        image, label  = read_data(filename_queue,flags)
        image = tf.cast(image, tf.float32)
        label = tf.cast(label, tf.float32)
        image, label = process_image_and_label(image, label, mean_image, variance_image,flags)

        # Generate a batch of images and labels by building up a queue of examples.
        image, label = generate_batch(image, label, min_queue_examples,
                                              int(flags['batch_size'] / flags['n_classes']),
                                              shuffle=False, flags=flags)
        combine_image_dict.append(image)
        combine_label_dict.append(label)

    out_image = tf.concat(0, combine_image_dict)
    out_label = tf.concat(0, combine_label_dict)

    return {'images':out_image, 'labels':out_label}