import tensorflow as tf
import numpy as np

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

    # conv1_1 128*128
    with tf.variable_scope('conv1_1') as scope:
        stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 3))
        kernel1_1 = tf.get_variable(name='weights', shape = [3, 3, 3, 64],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(images, kernel1_1, [1, 1, 1, 1], padding='SAME')
        biases1_1 = tf.get_variable(name='biases', shape=[64], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases1_1)
        conv1_1 = tf.nn.relu(out, name=scope.name)

    # conv1_2 128*128
    with tf.variable_scope('conv1_2') as scope:
        stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 64))
        kernel1_2 = tf.get_variable(name='weights', shape = [3, 3, 64, 64],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(conv1_1, kernel1_2, [1, 1, 1, 1], padding='SAME')
        biases1_2 = tf.get_variable(name='biases', shape=[64], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases1_2)
        conv1_2 = tf.nn.relu(out, name=scope.name)

    # pool1 128*128
    pool1 = tf.nn.max_pool(conv1_2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool1')

    # conv2_1 64*64
    with tf.variable_scope('conv2_1') as scope:
        stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 64))
        kernel2_1 = tf.get_variable(name='weights', shape=[3, 3, 64, 128],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(pool1, kernel2_1, [1, 1, 1, 1], padding='SAME')
        biases2_1 = tf.get_variable(name='biases', shape=[128], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases2_1)
        conv2_1 = tf.nn.relu(out, name=scope.name)

    # conv2_2 64*64
    with tf.variable_scope('conv2_2') as scope:
        stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 128))
        kernel2_2 = tf.get_variable(name='weights', shape=[3, 3, 128, 128],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(conv2_1, kernel2_2, [1, 1, 1, 1], padding='SAME')
        biases2_2 = tf.get_variable(name='biases', shape=[128], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases2_2)
        conv2_2 = tf.nn.relu(out, name=scope.name)

    # pool2 64*64
    pool2 = tf.nn.max_pool(conv2_2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool2')

    # conv3_1 32*32
    with tf.variable_scope('conv3_1') as scope:
        stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 128))
        kernel3_1 = tf.get_variable(name='weights', shape=[3, 3, 128, 256],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(pool2, kernel3_1, [1, 1, 1, 1], padding='SAME')
        biases3_1 = tf.get_variable(name='biases', shape=[256], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases3_1)
        conv3_1 = tf.nn.relu(out, name=scope.name)

    # conv3_2 32*32
    with tf.variable_scope('conv3_2') as scope:
        stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 256))
        kernel3_2 = tf.get_variable(name='weights', shape=[3, 3, 256, 256],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(conv3_1, kernel3_2, [1, 1, 1, 1], padding='SAME')
        biases3_2 = tf.get_variable(name='biases', shape=[256], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases3_2)
        conv3_2 = tf.nn.relu(out, name=scope.name)

    # # conv3_3
    # with tf.variable_scope('conv3_3') as scope:
    #     stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 256))
    #     kernel3_3 = tf.get_variable(name='weights', shape=[3, 3, 256, 256],
    #                     initializer=tf.truncated_normal_initializer(stddev=stddev))
    #     conv = tf.nn.conv2d(conv3_2, kernel3_3, [1, 1, 1, 1], padding='SAME')
    #     biases3_3 = tf.get_variable(name='biases', shape=[256], initializer=tf.constant_initializer(0.0))
    #     out = tf.nn.bias_add(conv, biases3_3)
    #     conv3_3 = tf.nn.relu(out, name=scope.name)

    # pool3 32*32
    pool3 = tf.nn.max_pool(conv3_2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool3')

    # conv4_1 16*16
    with tf.variable_scope('conv4_1') as scope:
        stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 256))
        kernel4_1 = tf.get_variable(name='weights', shape=[3, 3, 256, 512],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(pool3, kernel4_1, [1, 1, 1, 1], padding='SAME')
        biases4_1 = tf.get_variable(name='biases', shape=[512], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases4_1)
        conv4_1 = tf.nn.relu(out, name=scope.name)

    # conv4_2 16*16
    with tf.variable_scope('conv4_2') as scope:
        stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 512))
        kernel4_2 = tf.get_variable(name='weights', shape=[3, 3, 512, 512],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(conv4_1, kernel4_2, [1, 1, 1, 1], padding='SAME')
        biases4_2 = tf.get_variable(name='biases', shape=[512], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases4_2)
        conv4_2 = tf.nn.relu(out, name=scope.name)

    # # conv4_3
    # with tf.variable_scope('conv4_3') as scope:
    #     stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 512))
    #     kernel4_3 = tf.get_variable(name='weights', shape=[3, 3, 512, 512],
    #                     initializer=tf.truncated_normal_initializer(stddev=stddev))
    #     conv = tf.nn.conv2d(conv4_2, kernel4_3, [1, 1, 1, 1], padding='SAME')
    #     biases4_3 = tf.get_variable(name='biases', shape=[512], initializer=tf.constant_initializer(0.0))
    #     out = tf.nn.bias_add(conv, biases4_3)
    #     conv4_3 = tf.nn.relu(out, name=scope.name)

    # pool4 16*16
    pool4 = tf.nn.max_pool(conv4_2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool4')

    # conv5_1 8*8
    with tf.variable_scope('conv5_1') as scope:
        stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 512))
        kernel5_1 = tf.get_variable(name='weights', shape=[3, 3, 512, 512],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(pool4, kernel5_1, [1, 1, 1, 1], padding='SAME')
        biases5_1 = tf.get_variable(name='biases', shape=[512], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases5_1)
        conv5_1 = tf.nn.relu(out, name=scope.name)

    # conv5_2 8*8
    with tf.variable_scope('conv5_2') as scope:
        stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 512))
        kernel5_2 = tf.get_variable(name='weights', shape=[3, 3, 512, 512],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(conv5_1, kernel5_2, [1, 1, 1, 1], padding='SAME')
        biases5_2 = tf.get_variable(name='biases', shape=[512], initializer=tf.constant_initializer(0.0))
        out = tf.nn.bias_add(conv, biases5_2)
        conv5_2 = tf.nn.relu(out, name=scope.name)

    # # conv5_3
    # with tf.variable_scope('conv5_3') as scope:
    #     stddev = tf.sqrt(2 / tf.to_float(3 * 3 * 512))
    #     kernel5_3 = tf.get_variable(name='weights', shape=[3, 3, 512, 512],
    #                     initializer=tf.truncated_normal_initializer(stddev=stddev))
    #     conv = tf.nn.conv2d(conv5_2, kernel5_3, [1, 1, 1, 1], padding='SAME')
    #     biases5_3 = tf.get_variable(name='biases', shape=[512], initializer=tf.constant_initializer(0.0))
    #     out = tf.nn.bias_add(conv, biases5_3)
    #     conv5_3 = tf.nn.relu(out, name=scope.name)

    # pool5 8*8
    pool5 = tf.nn.max_pool(conv5_2,
                           ksize=[1, 2, 2, 1],
                           strides=[1, 2, 2, 1],
                           padding='SAME',
                           name='pool4')

    # fc1
    with tf.variable_scope('fc1') as scope:
        shape = int(np.prod(pool5.get_shape()[1:]))
        stddev = tf.sqrt(2 / tf.to_float(shape))
        fc1w = tf.get_variable(name='weights', shape=[shape, 1024],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        fc1b = tf.get_variable(name='biases', shape=[1024], initializer=tf.constant_initializer(0.0))
        pool5_flat = tf.reshape(pool5, [-1, shape])
        fc1l = tf.nn.bias_add(tf.matmul(pool5_flat, fc1w), fc1b)
        fc1 = tf.nn.relu(fc1l)
        fc1 = tf.nn.dropout(fc1, keep_prob)

    # fc2
    with tf.variable_scope('fc2') as scope:
        stddev = tf.sqrt(2 / tf.to_float(1024))
        fc2w = tf.get_variable(name='weights', shape=[1024, 1024],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        fc2b = tf.get_variable(name='biases', shape=[1024], initializer=tf.constant_initializer(0.0))
        fc2l = tf.nn.bias_add(tf.matmul(fc1, fc2w), fc2b)
        fc2 = tf.nn.relu(fc2l)
        fc2 = tf.nn.dropout(fc2, keep_prob)

    # fc3
    with tf.variable_scope('fc3') as scope:
        stddev = tf.sqrt(2 / tf.to_float(1024))
        fc3w = tf.get_variable(name='weights', shape=[1024, flags['n_classes']],
                        initializer=tf.truncated_normal_initializer(stddev=stddev))
        fc3b = tf.get_variable(name='biases', shape=[flags['n_classes']], initializer=tf.constant_initializer(0.0))
        fc3l = tf.nn.bias_add(tf.matmul(fc2, fc3w), fc3b)

    return tf.expand_dims(tf.expand_dims(tf.nn.softmax(fc3l), 1), 1), \
           {'kernel1_1': kernel1_1, 'biases1_1': biases1_1,
            'kernel1_2': kernel1_2, 'biases1_2': biases1_2,
            'kernel2_1': kernel2_1, 'biases2_1': biases2_1,
            'kernel2_2': kernel2_2, 'biases2_2': biases2_2,
            'kernel3_1': kernel3_1, 'biases3_1': biases3_1,
            'kernel3_2': kernel3_2, 'biases3_2': biases3_2,
            'kernel4_1': kernel4_1, 'biases4_1': biases4_1,
            'kernel4_2': kernel4_2, 'biases4_2': biases4_2,
            'kernel5_1': kernel5_1, 'biases5_1': biases5_1,
            'kernel5_2': kernel5_2, 'biases5_2': biases5_2,
            'fc1w': fc1w, 'fc1b': fc1b,
            'fc2w': fc2w, 'fc2b': fc2b,
            'fc3w': fc3w, 'fc3b': fc3b,
            'conv1_1': conv1_1, 'conv1_2': conv1_2, 'pool1': pool1,
            'conv2_1': conv2_1, 'conv2_2': conv2_2, 'pool2': pool2,
            'conv3_1': conv3_1, 'conv3_2': conv3_2, 'pool3': pool3,
            'conv4_1': conv4_1, 'conv4_2': conv4_2, 'pool4': pool4,
            'conv5_1': conv5_1, 'conv5_2': conv5_2, 'pool5': pool5,
            'fc1': fc1, 'fc2': fc2, 'fc3l': fc3l
            }
           # {'kernel1_1': kernel1_1, 'biases1_1': biases1_1,
           #  'kernel1_2': kernel1_2, 'biases1_2': biases1_2,
           #  'kernel2_1': kernel2_1, 'biases2_1': biases2_1,
           #  'kernel2_2': kernel2_2, 'biases2_2': biases2_2,
           #  'kernel3_1': kernel3_1, 'biases3_1': biases3_1,
           #  'kernel3_2': kernel3_2, 'biases3_2': biases3_2,
           #  'kernel3_3': kernel3_3, 'biases3_3': biases3_2,
           #  'kernel4_1': kernel4_1, 'biases4_1': biases4_1,
           #  'kernel4_2': kernel4_2, 'biases4_2': biases4_2,
           #  'kernel4_3': kernel4_3, 'biases4_3': biases4_3,
           #  'kernel5_1': kernel5_1, 'biases5_1': biases5_1,
           #  'kernel5_2': kernel5_2, 'biases5_2': biases5_2,
           #  'kernel5_3': kernel5_3, 'biases5_3': biases5_3,
           #  'fc1w': fc1w, 'fc1b': fc1b,
           #  'fc2w': fc2w, 'fc2b': fc2b,
           #  'fc3w': fc3w, 'fc3b': fc3b,
           #  'conv1_1':conv1_1, 'conv1_2':conv1_2, 'pool1':pool1,
           #  'conv2_1': conv2_1, 'conv2_2': conv2_2, 'pool2': pool2,
           #  'conv3_1': conv3_1, 'conv3_2': conv3_2, 'conv3_3': conv3_3, 'pool3': pool3,
           #  'conv4_1': conv4_1, 'conv4_2': conv4_2, 'conv4_3': conv4_3, 'pool4': pool4,
           #  'conv5_1': conv5_1, 'conv5_2': conv5_2, 'conv5_3': conv5_3, 'pool5': pool5,
           #  'fc1': fc1, 'fc2':fc2, 'fc3l':fc3l
           #  }

##############################################################################
def load_pretrain_model(sess,parameters,pretrain_path):
    weights = np.load(pretrain_path)

    sess.run(parameters['kernel1_1'].assign(weights['conv1_1_W']))
    sess.run(parameters['kernel1_2'].assign(weights['conv1_2_W']))
    sess.run(parameters['kernel2_1'].assign(weights['conv2_1_W']))
    sess.run(parameters['kernel2_2'].assign(weights['conv2_2_W']))
    sess.run(parameters['kernel3_1'].assign(weights['conv3_1_W']))
    sess.run(parameters['kernel3_2'].assign(weights['conv3_2_W']))
    sess.run(parameters['kernel3_3'].assign(weights['conv3_3_W']))
    sess.run(parameters['kernel4_1'].assign(weights['conv4_1_W']))
    sess.run(parameters['kernel4_2'].assign(weights['conv4_2_W']))
    sess.run(parameters['kernel4_3'].assign(weights['conv4_3_W']))
    sess.run(parameters['kernel5_1'].assign(weights['conv5_1_W']))
    sess.run(parameters['kernel5_2'].assign(weights['conv5_2_W']))
    sess.run(parameters['kernel5_3'].assign(weights['conv5_3_W']))
    sess.run(parameters['biases1_1'].assign(weights['conv1_1_b']))
    sess.run(parameters['biases1_2'].assign(weights['conv1_2_b']))
    sess.run(parameters['biases2_1'].assign(weights['conv2_1_b']))
    sess.run(parameters['biases2_2'].assign(weights['conv2_2_b']))
    sess.run(parameters['biases3_1'].assign(weights['conv3_1_b']))
    sess.run(parameters['biases3_2'].assign(weights['conv3_2_b']))
    sess.run(parameters['biases3_3'].assign(weights['conv3_3_b']))
    sess.run(parameters['biases4_1'].assign(weights['conv4_1_b']))
    sess.run(parameters['biases4_2'].assign(weights['conv4_2_b']))
    sess.run(parameters['biases4_3'].assign(weights['conv4_3_b']))
    sess.run(parameters['biases5_1'].assign(weights['conv5_1_b']))
    sess.run(parameters['biases5_2'].assign(weights['conv5_2_b']))
    sess.run(parameters['biases5_3'].assign(weights['conv5_3_b']))
    sess.run(parameters['fc1w'].assign(weights['fc6_W']))
    sess.run(parameters['fc1b'].assign(weights['fc6_b']))
    sess.run(parameters['fc2w'].assign(weights['fc7_W']))
    sess.run(parameters['fc2b'].assign(weights['fc7_b']))
    # sess.run(parameters['fc3w'].assign(weights['fc8_W']))
    # sess.run(parameters['fc3b'].assign(weights['fc8_b']))

##############################################################################
def loss(softmax, labels, curr_epoch , flags):
    # sigmoid loss
    labels = tf.squeeze(labels)
    labels = tf.cast(labels, tf.int64)
    labels = tf.expand_dims(labels, dim=1)
    onehot = tf.one_hot(labels, depth=flags['n_classes'], on_value=1.0, off_value=0.0, axis=2)
    onehot = tf.expand_dims(onehot, dim=1)

    epsilon = 1e-6
    truncated_softmax = tf.clip_by_value(softmax, epsilon, 1.0 - epsilon)
    cross_entropy_log_loss = -tf.reduce_sum(onehot * tf.log(truncated_softmax), reduction_indices=[3], keep_dims=True)
    avg_cross_entropy_log_loss = tf.reduce_mean(cross_entropy_log_loss, reduction_indices=[0, 1, 2])

    return {'truncated_softmax': truncated_softmax,
            'cross_entropy_log_loss': cross_entropy_log_loss,
            'avg_cross_entropy_log_loss': avg_cross_entropy_log_loss,
            'labels': labels,
            'onehot': onehot}

##############################################################################
def train(total_loss, global_step, parameters, flags):
    optimizer = tf.train.AdamOptimizer(learning_rate=flags['initial_learning_rate'])
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
