import os
import math
import tensorflow as tf
import collections
import numpy as np

from KS_lib.general import KScsv


#####################################################################################
def read_data(filename_queue, flags):
    #keys = ['HE', 'DAPI', 'label']
    key_list =list(flags['dict_path'].keys())
    key_list.remove('group')
    
    queue_dict = {}  
    label_dict = {} 
 
    for ind,key in enumerate(key_list): 
         image_content = tf.read_file(filename_queue[ind])
         image = tf.image.decode_image(image_content,channels =3)
         image.set_shape(flags['size_input_patch'])
         queue_dict[key] = tf.cast(image,tf.float32)

         #filename_tensor = filename_queue.dequeue()
         splt = tf.string_split([filename_queue[ind]],'/').values         
         splt2 = tf.string_split([splt[-1]],'_').values         
         label = tf.string_to_number(splt2[0])
         print(label)
         label_dict[key] = label 

    return queue_dict,label_dict 


#####################################################################################
def process_image_and_label(queue_dict, mat_contents, flags):
    
    # Randomly flip the image.
    #keys = ['HE', 'DAPI', 'label']
    key_list =list(flags['dict_path'].keys())
    key_list.remove('group')
    
    r_flip = tf.random_uniform([2], 0, 1.0, dtype=tf.float32)

    for key in key_list:

        # left right flip
        img = queue_dict[key]
        img1 = tf.cond(tf.less(r_flip[0], 0.5), lambda: tf.reverse(img, [1]), lambda: img)

        # rotate

        #rotate 90 
        cond90 = tf.less(r_flip[1], 0.25) 
        img2 = tf.cond(cond90, lambda: tf.contrib.image.rotate(img1, math.pi/4), lambda: img1)

        #rotate 180 
        cond180 = tf.logical_and(tf.greater(r_flip[1], 0.25), tf.less(r_flip[1], 0.5)) 
        img2 = tf.cond(cond180, lambda: tf.contrib.image.rotate(img1, math.pi/2), lambda: img1)

        #rotate 270 
        cond270 = tf.logical_and(tf.greater(r_flip[1], 0.5), tf.less(r_flip[1], 0.75)) 
        img2 = tf.cond(cond270, lambda: tf.contrib.image.rotate(img1, 3*math.pi/4), lambda: img1)


        queue_dict[key] = img 
       
    #label = tf.cond(tf.less(r_flip[0], 0.5), lambda: tf.reverse(label, [1]), lambda: label)
    #weight = tf.cond(tf.less(r_flip[0], 0.5), lambda: tf.reverse(weight, [1]), lambda: weight)

    # up down
    #mirror = tf.less(tf.stack([r_flip[1], 1.0, 1.0]), 0.5)
    #image = tf.cond(tf.less(r_flip[0], 0.5), lambda: tf.reverse(image, [0]), lambda: image)
    #label = tf.cond(tf.less(r_flip[0], 0.5), lambda: tf.reverse(label, [0]), lambda: label)
    #weight = tf.cond(tf.less(r_flip[0], 0.5), lambda: tf.reverse(weight, [0]), lambda: weight)

    # transpose
    #mirror = tf.less(tf.stack([r_flip[2], 1.0 - r_flip[2]]), 0.5)
    #mirror = tf.cast(mirror, tf.int32)
    #mirror = tf.stack([mirror[0], mirror[1], 2])

    #image = tf.transpose(image, perm=mirror)
    #label = tf.transpose(label, perm=mirror)
    #weight = tf.transpose(weight, perm=mirror)

    # Because these operations are not commutative, consider randomizing
    # the order their operation.
    # image = tf.image.random_brightness(image, max_delta=63)
    # image = tf.image.random_contrast(image, lower=0.2, upper=1.8)
    # image = tf.image.random_hue(image, max_delta=0.2)
    # image = tf.image.random_saturation(image, lower=0.5, upper=1.5)

    # random brightness field
    # M = flags['size_input_patch'][0]*10
    # N = flags['size_input_patch'][1]*10
    # r0 = np.random.uniform(0, M, 10*10)
    # r1 = np.random.uniform(0, N, 10*10)
    #
    # r0 = np.unique(np.rint(np.append(0, np.append(r0, M)))).astype(np.int)
    # r1 = np.unique(np.rint(np.append(0, np.append(r1, N)))).astype(np.int)
    #
    # random_field_mat = np.zeros(shape=(M, N, 1), dtype=np.float32)
    # for i in xrange(len(r0) - 1):
    #     for j in xrange(len(r1) - 1):
    #         random_field_mat[r0[i]:r0[i + 1], r1[j]:r1[j + 1], :] = np.random.uniform(0.5, 1.5, 1)
    #
    # big_random_field = tf.convert_to_tensor(random_field_mat, dtype = tf.float32)
    # random_field = tf.random_crop(big_random_field, [flags['size_input_patch'][0],
    #                                                  flags['size_input_patch'][1],
    #                                                  1])
    #
    # random_field = tf.image.random_flip_up_down(random_field)
    # random_field = tf.image.random_flip_left_right(random_field)

    # hsv = tf.image.rgb_to_hsv(image)
    # h, s, v = tf.split(2, 3, hsv)
    # v = v*random_field
    # hsv = tf.concat(2, [h, s, v])
    # image = tf.image.hsv_to_rgb(hsv)

    #######################################################################

    # random gamma adjustment
    #r, g, b = tf.split(image, 3, 2)
    #rr = tf.random_uniform([1], minval=np.log(0.25), maxval=np.log(4), dtype=tf.float32)
    #rb = tf.random_uniform([1], minval=np.log(0.25), maxval=np.log(4), dtype=tf.float32)
    #r = 255.0 * tf.pow(r / 255.0, tf.exp(rr))
    #b = 255.0 * tf.pow(b / 255.0, tf.exp(rb))
    #image = tf.concat([r, g, b], 2)

    # random brightness
    #image = tf.image.random_brightness(image, max_delta=63)

    # random contrast
    #image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    # random saturation
    #image = tf.image.random_saturation(image, lower=0.8, upper=1.2)

    # random hue
    # image = tf.image.random_hue(image, max_delta=0.4)

    # normalise a patch to have zero mean and a unit variance
    for ind,key in enumerate(key_list):
        mean_img = mat_contents[key + '_mean']
        variance_img = mat_contents[key + '_var']

        queue_dict[key].set_shape(flags['size_input_patch'])
    
        epsilon = 1e-6
        queue_dict[key] = queue_dict[key] - mean_img
        queue_dict[key] = queue_dict[key] / tf.sqrt(variance_img + epsilon)

        #weight = weight / 255.0

    return queue_dict 


#####################################################################################
def generate_batch(queue_dict, label_dict, min_queue_examples, batch_size, shuffle, flags):
    # create list of lists
    queue_list = []
    label_list = [] 
    for key,value in queue_dict.items():
        #print(value)
        queue_list.append(value)
    # append single label after, since labels are identical for all types of images 
    queue_list.append(label_dict['HE']) 
  
    #print(queue_list) 
    num_preprocess_threads = flags['num_preprocess_threads']
    if shuffle:
        out_list = tf.train.shuffle_batch(
            queue_list,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size,
            min_after_dequeue=min_queue_examples)
    else:
        out_list = tf.train.batch(
            queue_list,
            batch_size=batch_size,
            num_threads=num_preprocess_threads,
            capacity=min_queue_examples + 3 * batch_size)

    # outlist - list of 4 elements, 3 of size 32,512,512,3, last one of size 32
    out_img = out_list[:-1]
    out_label = out_list[-1:]
    return out_img, out_label 



#####################################################################################
def inputs(object_folder, mode, flags, mat_contents):
    #keys = ['HE', 'DAPI', 'label']
    key_list =list(flags['dict_path'].keys())
    key_list.remove('group')

    if mode == 'train':
        log_file_path = os.path.join(object_folder, 'train', 'train_log.csv')
    else:
        log_file_path = os.path.join(object_folder, 'val', 'val_log.csv')

    log_list = KScsv.read_csv(log_file_path)

    #image_dict = collections.defaultdict(list)
    #label_dict = collections.defaultdict(list)
    #weight_dict = collections.defaultdict(list)

    all_img_dict = {}
    slice_input_list = []

    for ind,key in enumerate(key_list):
        key_img_list = []       
 
        #deal with mean, var img         
        mean_img = np.float32(mat_contents[key + '_mean'])
        var_img = np.float32(mat_contents[key + '_var'])

        if mean_img.ndim == 2:
            mean_img = np.expand_dims(mean_img, axis=2)
        if var_img.ndim == 2:
            var_img = np.expand_dims(var_img, axis=2)
        
        #mean_image = tf.constant(mean_img, name='mean_image')
        #var_image = tf.constant(var_img, name='var_image')

        for row in log_list:
            key_img_list.append(row[ind])

        slice_input_list.append(key_img_list)
    
    min_queue_examples = int(len(slice_input_list[0]) * flags['min_fraction_of_examples_in_queue'])
    print('Filling queue with %d images before starting to train. '
    'This will take a few minutes.' % min_queue_examples)
    
    # Create a queue that produces the filenames to read.
    combine_image_dict = list()

    #fliename_queue- list of file names
    filename_queue = tf.train.slice_input_producer(slice_input_list,shuffle = True)
    queue_dict,label_dict = read_data(filename_queue, flags)
    processed_dict = process_image_and_label(queue_dict, mat_contents, flags)

        # Generate a batch of images and labels by building up a queue of examples.
    batch_list, label = generate_batch(processed_dict, label_dict, min_queue_examples,
                                          int(flags['batch_size']), shuffle=False, flags=flags)

    
    #create final combined dict    
    #combine_image_dict = collections.defaultdict(list)
    #combine_label_dict = collections.defaultdict(list)
   
    combine_image_dict = {}
    combine_label_dict = {}
    for ind,key in enumerate(key_list):
        #combine_image_dict[key].append(batch_list[ind])
        combine_image_dict[key] = batch_list[ind]
        #combine_image_dict[key] = tf.concat(combine_image_dict[key],0)

        combine_label_dict[key] = label

    return combine_image_dict, combine_label_dict 
	
#####################################################################################
def inputs2(object_folder, mode, flags,mat_contents):
    if mode == 'train':
        log_file_path = os.path.join(object_folder, 'train', 'train_log.csv')
    else:
        log_file_path = os.path.join(object_folder, 'val', 'val_log.csv')

    log_list = KScsv.read_csv(log_file_path)
    
    #key_list = ['HE', 'DAPI', 'weight']
    key_list = list(flags['dict_path'].keys())[:-1] # all items except last item: 'group'
	
    allimageslist = []
    for ind,key in enumerate(key_list):
        image_dict = collections.defaultdict(list)
        #label_dict = collections.defaultdict(list)
        #weight_dict = collections.defaultdict(list)

        for row in log_list:
            image_dict[key].append(row[ind])
            #label_dict['label'].append(row[1])
            #weight_dict['weight'].append(row[2])

        min_queue_examples = int(len(image_dict[key[0]]) * flags['min_fraction_of_examples_in_queue'])
        print('Filling queue with %d images before starting to train. '
              'This will take a few minutes.' % min_queue_examples)

        # Create a queue that produces the filenames to read.
        combine_image_dict = list()
        #combine_label_dict = list()
        #combine_weight_dict = list()
        allimageslist.append(image_dict[key])

        filename_queue = tf.train.slice_input_producer(allimageslist,shuffle=True)
        queue_dict = read_data(filename_queue, flags)
        image, label, weight = process_image_and_label(image, label, weight, mean_image, variance_image, flags)

        # Generate a batch of images and labels by building up a queue of examples.
        image, label, weight = generate_batch(image, label, weight, min_queue_examples,
                                          int(flags['batch_size']), shuffle=False, flags=flags)
        combine_image_dict.append(image)
        combine_label_dict.append(label)
        combine_weight_dict.append(weight)

        out_image = tf.concat(combine_image_dict, 0)
        out_label = tf.concat(combine_label_dict, 0)
        out_weight = tf.concat(combine_weight_dict, 0)

    return {'images': out_image, 'labels': out_label, 'weights': out_weight}
