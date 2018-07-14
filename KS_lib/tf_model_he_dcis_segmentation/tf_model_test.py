import os
import time
from itertools import izip

import cv2
import numpy as np
import tensorflow as tf

from KS_lib.general import matlab
from KS_lib.image import KSimage
from KS_lib.prepare_data import routine
from KS_lib.tf_model_he_dcis_segmentation import tf_model
from KS_lib.tf_model_he_dcis_segmentation import tf_model_input_test

########################################################################################################################
def batch_processing(filename, sess, logits_test, parameters, images_test, keep_prob,
                     mean_image, variance_image, flags):
    # Read image and extract patches
    patches, image_size, nPatches, ori_dim = tf_model_input_test.read_data_test(filename, flags)

    def batches(generator, size):
        source = generator
        while True:
            chunk = [val for _, val in izip(xrange(size), source)]
            if not chunk:
                raise StopIteration
            yield chunk

    # Construct batch indices
    batch_index = range(0, nPatches, flags['test_batch_size'])
    if nPatches not in batch_index:
        batch_index.append(nPatches)

    # Process all_patches
    shape = np.hstack([nPatches, flags['size_output_patch']])
    shape[-1] = logits_test.get_shape()[3].value
    all_patches = np.zeros(shape, dtype=np.float32)

    for ipatch, chunk in enumerate(batches(patches, flags['test_batch_size'])):
        start_idx = batch_index[ipatch]
        end_idx = batch_index[ipatch + 1]

        temp = tf_model_input_test.inputs_test(chunk, mean_image, variance_image)

        if temp.shape[0] < flags['test_batch_size']:
            rep = np.tile(temp[-1, :, :, :], [flags['test_batch_size'] - temp.shape[0], 1, 1, 1])
            temp = np.vstack([temp, rep])

        pred = sess.run(logits_test, feed_dict={images_test: temp, keep_prob: 1.0})
        all_patches[start_idx:end_idx, :, :, :] = pred[range(end_idx - start_idx), :, :, :]

    result = tf_model_input_test.MergePatches_test(all_patches, flags['stride_test'],
                                                   image_size, flags['size_input_patch'],
                                                   flags['size_output_patch'], flags)
    result = result*255.0
    result = result.astype(np.uint8)
    result = KSimage.imresize(result,4.0)
    result = np.argmax(result, axis=2)

    # resize may not preserve the original dimensions of the image
    # append with zero or remove excessive pixels in each dimension
    if result.shape[0] < ori_dim[0]:
        result = np.pad(result, ((0, ori_dim[0] - result.shape[0]), (0, 0)), 'constant', constant_values=0)
    else:
        result = result[0:ori_dim[0], :]

    if result.shape[1] < ori_dim[1]:
        result = np.pad(result, ((0, 0), (0, ori_dim[1] - result.shape[1])), 'constant', constant_values=0)
    else:
        result = result[:, 0:ori_dim[1]]

    mask = result == 1
    mask = mask.astype(np.uint8) * 255
    im, contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    temp_mask = np.zeros(mask.shape[:2], dtype='uint8')
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500 ** 2:
            cv2.drawContours(temp_mask, [cnt], -1, 255, -1)

    result = temp_mask

    return result

########################################################################################################################
def test(object_folder, model_path, filename_list, result_path ,flags, igpu):
    checkpoint_dir = os.path.join(object_folder, 'checkpoint')
    mat_contents = matlab.load(os.path.join(checkpoint_dir, 'network_stats.mat'))
    mean_image = np.float32(mat_contents['mean_image'])
    variance_image = np.float32(mat_contents['variance_image'])

    ###########################################################
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = igpu
    ###########################################################

    with tf.Graph().as_default(), tf.device(flags['gpu']):
        keep_prob = tf.placeholder(tf.float32)
        # Place holder for patches
        images_test = tf.placeholder(tf.float32, shape=(np.hstack([flags['test_batch_size'], flags['size_input_patch']])))
        # Network
        with tf.variable_scope("network") as scope:
            logits_test, parameters = tf_model.inference(images_test, keep_prob, flags)
        # Saver and initialisation
        saver = tf.train.Saver()
        init = tf.initialize_all_variables()

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=flags['gpu_memory_fraction'])
        config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

        with tf.Session(config = config) as sess:
            # Initialise and load variables
            sess.run(init)
            saver.restore(sess, model_path)

            result_dir = result_path
            routine.create_dir(result_dir)

            for iImage, file in enumerate(filename_list):
                start_time = time.time()
                file = file[0]
                basename = os.path.basename(file)
                basename = os.path.splitext(basename)[0]
                savename = os.path.join(result_dir, basename + '.png')
                if not os.path.exists(savename):
                    result = batch_processing(file, sess, logits_test, parameters, images_test,
                                              keep_prob, mean_image, variance_image, flags)
                    # matlab.save(savename,{'mask':result})
                    KSimage.imwrite(result,savename)
                duration = time.time() - start_time
                print('Finish segmenting DCIS regions on the H&E image of sample %d out of %d samples (%.2f sec)' %
                      (iImage + 1,len(filename_list),duration))

########################################################################################################################
