import tensorflow as tf
from KS_lib.tf_op import layer
from sklearn import metrics
import numpy as np
import os


##############################################################################
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

##############################################################################
def inference(images, keep_prob, flags):
    # conv1_1 51x51
    with tf.variable_scope('conv1_1') as scope:
        stddev = tf.sqrt(2 / tf.to_float(2 * 2 * 3))
        kernel1_1 = tf.get_variable(name='weights', shape = [2, 2, 3, 64],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(images, kernel1_1, [1, 1, 1, 1], padding='VALID')
        biases1_1 = tf.get_variable(name='biases', shape=[64], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases1_1)
        conv1_1 = tf.nn.relu(out, name=scope.name)

    # pool1 50x50
    pool1 = tf.nn.max_pool(conv1_1,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool1')

    # conv2_1 25x25
    with tf.variable_scope('conv2_1') as scope:
        stddev = tf.sqrt(2 / tf.to_float(2 * 2 * 64))
        kernel2_1 = tf.get_variable(name='weights', shape=[2, 2, 64, 128],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(pool1, kernel2_1, [1, 1, 1, 1], padding='VALID')
        biases2_1 = tf.get_variable(name='biases', shape=[128], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases2_1)
        conv2_1 = tf.nn.relu(out, name=scope.name)

    # pool2 24x24
    pool2 = tf.nn.max_pool(conv2_1,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool2')

    # conv3_1 12x12x128
    with tf.variable_scope('conv3_1') as scope:
        stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 128))
        kernel3_1 = tf.get_variable(name='weights', shape=[3, 3, 128, 256],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(pool2, kernel3_1, [1, 1, 1, 1], padding='VALID')
        biases3_1 = tf.get_variable(name='biases', shape=[256], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases3_1)
        conv3_1 = tf.nn.relu(out, name=scope.name)

    # pool3 10x10x256
    pool3 = tf.nn.max_pool(conv3_1,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='VALID',
                           name='pool3')

    # fc1 5x5
    with tf.variable_scope('fc1') as scope:
        shape = int(np.prod(pool3.get_shape()[1:]))
        stddev = tf.sqrt(2 / tf.to_float(shape))
        fc1w = tf.get_variable(name='weights', shape=[shape, 2048],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        fc1b = tf.get_variable(name='biases', shape=[2048], initializer=tf.constant_initializer(0.0))
        pool3_flat = tf.reshape(pool3, [-1, shape])
        fc1l = tf.nn.bias_add(tf.matmul(pool3_flat, fc1w), fc1b)
        fc1 = tf.nn.relu(fc1l)
        fc1 = tf.nn.dropout(fc1, keep_prob)

    # fc2
    with tf.variable_scope('fc2') as scope:
        stddev = tf.sqrt(2 / tf.to_float(2048))
        fc2w = tf.get_variable(name='weights', shape=[2048, 2048],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        fc2b = tf.get_variable(name='biases', shape=[2048], initializer=tf.constant_initializer(0.0))
        fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
        fc2 = tf.nn.relu(fc2l)
        fc2 = tf.nn.dropout(fc2, keep_prob)

    # fc3
    with tf.variable_scope('fc3') as scope:
        stddev = tf.sqrt(2 / tf.to_float(2048))
        fc3w = tf.get_variable(name='weights', shape=[2048, flags['size_output_patch'][0] * flags['size_output_patch'][1]],
                               initializer=tf.truncated_normal_initializer(stddev=stddev))
        fc3b = tf.get_variable(name='biases', shape=[flags['size_output_patch'][0] * flags['size_output_patch'][1]], initializer=tf.constant_initializer(0.0))
        fc3l = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)
        fc3 = tf.sigmoid(fc3l)
        fc3 = tf.reshape(fc3, [int(fc3.get_shape()[0]), flags['size_output_patch'][0], flags['size_output_patch'][1], flags['size_output_patch'][2]])


    return fc3, \
           {'kernel1_1': kernel1_1, 'biases1_1': biases1_1,
            'kernel2_1': kernel2_1, 'biases2_1': biases2_1,
            'kernel3_1': kernel3_1, 'biases3_1': biases3_1,
            'fc1w': fc1w, 'fc1b': fc1b,
            'fc2w': fc2w, 'fc2b': fc2b,
            'fc3w': fc3w, 'fc3b': fc3b,
            'conv1_1': conv1_1, 'pool1': pool1,
            'conv2_1': conv2_1, 'pool2': pool2,
            'conv3_1': conv3_1, 'pool3': pool3,
            'fc1': fc1, 'fc2': fc2, 'fc3l': fc3l, 'fc3':fc3
            }

##############################################################################
# def load_pretrain_model(sess,parameters,pretrain_path):
    # weights = np.load(pretrain_path)
    #
    # sess.run(parameters['kernel1_1'].assign(weights['conv1_1_W']))
    # sess.run(parameters['kernel1_2'].assign(weights['conv1_2_W']))
    # sess.run(parameters['kernel2_1'].assign(weights['conv2_1_W']))
    # sess.run(parameters['kernel2_2'].assign(weights['conv2_2_W']))
    # sess.run(parameters['kernel3_1'].assign(weights['conv3_1_W']))
    # sess.run(parameters['kernel3_2'].assign(weights['conv3_2_W']))
    # sess.run(parameters['kernel3_3'].assign(weights['conv3_3_W']))
    # sess.run(parameters['kernel4_1'].assign(weights['conv4_1_W']))
    # sess.run(parameters['kernel4_2'].assign(weights['conv4_2_W']))
    # sess.run(parameters['kernel4_3'].assign(weights['conv4_3_W']))
    # sess.run(parameters['kernel5_1'].assign(weights['conv5_1_W']))
    # sess.run(parameters['kernel5_2'].assign(weights['conv5_2_W']))
    # sess.run(parameters['kernel5_3'].assign(weights['conv5_3_W']))
    # sess.run(parameters['biases1_1'].assign(weights['conv1_1_b']))
    # sess.run(parameters['biases1_2'].assign(weights['conv1_2_b']))
    # sess.run(parameters['biases2_1'].assign(weights['conv2_1_b']))
    # sess.run(parameters['biases2_2'].assign(weights['conv2_2_b']))
    # sess.run(parameters['biases3_1'].assign(weights['conv3_1_b']))
    # sess.run(parameters['biases3_2'].assign(weights['conv3_2_b']))
    # sess.run(parameters['biases3_3'].assign(weights['conv3_3_b']))
    # sess.run(parameters['biases4_1'].assign(weights['conv4_1_b']))
    # sess.run(parameters['biases4_2'].assign(weights['conv4_2_b']))
    # sess.run(parameters['biases4_3'].assign(weights['conv4_3_b']))
    # sess.run(parameters['biases5_1'].assign(weights['conv5_1_b']))
    # sess.run(parameters['biases5_2'].assign(weights['conv5_2_b']))
    # sess.run(parameters['biases5_3'].assign(weights['conv5_3_b']))
    # sess.run(parameters['fc1w'].assign(weights['fc6_W']))
    # sess.run(parameters['fc1b'].assign(weights['fc6_b']))
    # sess.run(parameters['fc2w'].assign(weights['fc7_W']))
    # sess.run(parameters['fc2b'].assign(weights['fc7_b']))
    # sess.run(parameters['fc3w'].assign(weights['fc8_W']))
    # sess.run(parameters['fc3b'].assign(weights['fc8_b']))

##############################################################################
def loss(sigmoid, labels, weights, curr_epoch):
    epsilon = 1e-6

    #######################################################
    # number of postives and negatives
    beta = tf.to_float(labels > 0.25)
    n_pos = tf.to_float(tf.reduce_sum(beta, [0, 1, 2, 3]))
    n_neg = tf.to_float(tf.size(labels)) - n_pos

    #######################################################
    # class imbalance weight
    class_weights = beta

    class_weights = tf.cond(n_neg > n_pos, lambda: class_weights * (n_neg - n_pos) + n_pos,
                            lambda: (1.0 - class_weights) * (n_pos - n_neg) + n_neg)
    class_weights = class_weights / (n_pos + n_neg)

    #######################################################
    # final weight
    # total_weights = (class_weights + labels)
    # total_weights = 50.0*(labels + sigmoid)
    total_weights = (labels + 0.3 + weights)
    # total_weights = (class_weights)
    #######################################################
    # sigmoid loss
    sigmoid = tf.clip_by_value(sigmoid, epsilon, 1.0 - epsilon)
    cross_entropy_log_loss = - labels * tf.log(sigmoid) - (1 - labels) * tf.log(1 - sigmoid)
    cross_entropy_log_loss = (total_weights) * cross_entropy_log_loss
    avg_cross_entropy_log_loss = tf.reduce_mean(cross_entropy_log_loss, reduction_indices=[0, 1, 2])

    return {'truncated_softmax': sigmoid,
            'cross_entropy_log_loss': cross_entropy_log_loss,
            'avg_cross_entropy_log_loss': avg_cross_entropy_log_loss,
            'labels': labels,
            'class_weights':class_weights}

##############################################################################
def train(total_loss, global_step, parameters, flags):
    optimizer = tf.train.AdamOptimizer(learning_rate=flags['initial_learning_rate'],
                                       epsilon=1e-6)
    # grads_and_vars = optimizer.compute_gradients(total_loss)
    # train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    # var_list = [parameters['kernel1_1'], parameters['biases1_1'], parameters['fc3w'], parameters['fc3b']]
    # var_list = [parameters['fc3w'], parameters['fc3b']]
    # train_op = optimizer.minimize(total_loss, global_step = global_step,var_list = var_list)
    train_op = optimizer.minimize(total_loss, global_step=global_step)

    return train_op

##############################################################################
def accuracy(predicts, labels):
    predicts = tf.reshape(predicts, [-1])
    labels = tf.reshape(labels, [-1])

    TP = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(predicts, 1), tf.equal(labels, 1))))
    FP = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(predicts, 1), tf.equal(labels, 0))))
    FN = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(predicts, 0), tf.equal(labels, 1))))
    TN = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(predicts, 0), tf.equal(labels, 0))))

    precision = TP / tf.to_float(TP + FP)
    recall = TP / tf.to_float(TP + FN)
    f1score = 2 * precision * recall / tf.to_float(precision + recall)

    return {'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': TN,
            'precision': precision,
            'recall': recall,
            'f1score': f1score}
