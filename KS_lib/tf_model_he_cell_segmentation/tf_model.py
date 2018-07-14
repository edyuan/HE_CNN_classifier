import tensorflow as tf
from KS_lib.general import matlab
from KS_lib.tf_op import layer
from sklearn import metrics
import numpy as np
import math

##############################################################################
def inference(images, keep_prob, flags):

    # reg, green, blue = tf.split(3, 3, images)
    # images = tf.tile(blue, [1, 1, 1, 3])

    # images - 32,512,512,3 
    with tf.variable_scope('down1'):
        with tf.variable_scope('conv1relu1'):
            down1_conv1 = layer.down_conv_relu_valid(images, [3, 3], flags['size_input_patch'][2], 16)
        # after above - 32,510,510,16
        with tf.variable_scope('pool'):
            down1_pool = tf.nn.max_pool(down1_conv1, ksize = [1,3,3,1], strides=[1,3,3,1],padding='VALID')
        with tf.variable_scope('batchnorm'):
            down1_bn = tf.layers.batch_normalization(down1_pool,training=True)
        # after above - 32,170,170,16

    # 32,170,170,16 
    with tf.variable_scope('down2'):
        with tf.variable_scope('conv2relu2'):
            down2_conv1 = layer.down_conv_relu_valid(down1_bn, [3, 3], 16, 32)
        # after above - 32,168,168,32
        with tf.variable_scope('pool'):
            down2_pool = tf.nn.max_pool(down2_conv1, ksize = [1,2,2,1], strides=[1,2,2,1], padding='VALID')
        # after above - 32,84,84,32
        with tf.variable_scope('batchnorm'):
            down2_bn = tf.layers.batch_normalization(down2_pool,training=True)

    # 18x18
    with tf.variable_scope('down3'):
        with tf.variable_scope('conv3relu3'):
            down3_conv1 = layer.down_conv_relu_valid(down2_bn, [2, 2], 32, 64)
        # after above 32,83,83,64
        with tf.variable_scope('pad'):
            pad3 = tf.pad(down3_conv1, [[0,0],[2,2],[2,2],[0,0]])
        # after above 32,87,87,64
        with tf.variable_scope('pool'):
            down3_pool = tf.nn.max_pool(pad3, ksize = [1,2,2,1], strides=[1,2,2,1], padding='VALID')
        # after above 32, 43, 43, 64
        with tf.variable_scope('batchnorm'):
            down3_bn = tf.layers.batch_normalization(down3_pool,training=True)

    # 18x18
    with tf.variable_scope('down4'):
        with tf.variable_scope('conv4relu4'):
            down4_conv1 = layer.down_conv_relu_valid(down3_bn, [2, 2], 64, 64)
        # after above 32,42,42,64
        with tf.variable_scope('pad'):
            pad4 = tf.pad(down4_conv1, [[0,0],[2,2],[2,2],[0,0]])
        # after above 32,46,46,64
        with tf.variable_scope('pool'):
            down4_pool = tf.nn.max_pool(pad4, ksize = [1,3,3,1], strides=[1,3,3,1], padding='VALID')
        # after above 32,15,15,64   
        with tf.variable_scope('batchnorm'):
            down4_bn = tf.layers.batch_normalization(down4_pool,training=True)
 
    # x
    with tf.variable_scope('down5'):
        with tf.variable_scope('conv5relu5'):
            down5_conv1 = layer.down_conv_relu_valid(down4_bn, [3, 3], 64, 32)
        with tf.variable_scope('pool'):
            down5_pool = tf.nn.max_pool(down5_conv1, ksize = [1,3,3,1], strides=[1,3,3,1], padding='VALID')
        #after above 32,13,13,32 
        with tf.variable_scope('batchnorm'):
            down5_bn = tf.layers.batch_normalization(down5_pool,training=True)
    # FC
    with tf.variable_scope('fc'):
        reshapedvec = tf.reshape(down5_bn, [flags['batch_size'], -1 ]) 
        # after reshape 32,5408
        with tf.variable_scope('fc1'):
            fc1 = tf.contrib.layers.fully_connected(reshapedvec,256,activation_fn=tf.nn.relu) #by default xavier init for w
            # after above 32,256
        with tf.variable_scope('fc2'):
            fc2 = tf.contrib.layers.fully_connected(fc1,128,activation_fn=tf.nn.relu) #by default xavier init for w
            #fc2 = tf.Print(fc2, [fc2], summarize=32)
            # after above 32,128 
        with tf.variable_scope('softmax'): 
            softmax = tf.contrib.layers.fully_connected(fc2,4,activation_fn=None) #by default xavier init for w
            #softmax = tf.Print(softmax, [softmax], summarize = 32*4)
            softmax = tf.nn.softmax(softmax)

    return softmax, \
           {'softmax':softmax}

##############################################################################
def loss(softmax, labels, curr_epoch , flags, counts):
    #softmax = tf.Print(softmax, [softmax], summarize = 32*4)
    # softmax - 32,4
    # labels - dictionary - 32

    # number of postives and negatives
    #pos_mask = tf.to_float(tf.equal(labels,1))
    #neg_mask = tf.to_float(tf.equal(labels,0))
    #bd_mask = tf.to_float(tf.equal(labels,2))

    #n_pos = tf.to_float(tf.reduce_sum(pos_mask, [0, 1, 2, 3]))
    #n_neg = tf.to_float(tf.reduce_sum(neg_mask, [0, 1, 2, 3]))
    #n_bd = tf.to_float(tf.reduce_sum(bd_mask, [0, 1, 2, 3]))

    #######################################################
    # class imbalance weight

    #max_val = tf.reduce_max(tf.stack([n_pos,n_neg,n_bd]))

    #pos_weights = (pos_mask/n_pos)*max_val*1.0
    #neg_weights = (neg_mask/n_neg)*max_val*1.0
    #bd_weights = (bd_mask/n_bd)*max_val*1.0

    #class_weights = pos_weights + neg_weights + bd_weights
    #total_weights = class_weights + flags['alpha'] * weights

    # sigmoid loss
    labels_ = labels['HE']
    labels_ = tf.cast(labels_, tf.int64)
    # labels = tf.expand_dims(labels, dim=1)
    # onehot - 32,4
    onehot = tf.squeeze(tf.one_hot(labels_, depth=flags['n_classes'], on_value=1.0, off_value=0.0))
    # onehot = tf.squeeze(onehot, dim=3)

    #find weighting for different classes
    weights = 1 / flags['n_classes'] * np.sum(counts['counts']) / counts['counts']  
    weights = np.array(weights)
    weights[weights == math.inf] = 0    

    epsilon = 1e-8
    truncated_softmax = tf.clip_by_value(softmax, epsilon, 1.0 - epsilon)
    softmax = tf.Print(softmax, [softmax], summarize = 32*4, message="SM: ")
    cross_entropy_log_loss = -tf.reduce_sum( weights*onehot * tf.log(truncated_softmax), reduction_indices=[1])
    #cross_entropy_log_loss = tf.nn.softmax_cross_entropy_with_logits(labels = onehot, logits = softmax)
    cross_entropy_log_loss = tf.Print(cross_entropy_log_loss, [cross_entropy_log_loss], summarize=32, message="CE: ")
    #cross_entropy_log_loss = (total_weights) * cross_entropy_log_loss
    avg_cross_entropy_log_loss = tf.reduce_mean(cross_entropy_log_loss, reduction_indices=[0])

    return {'truncated_softmax': truncated_softmax,
            'cross_entropy_log_loss': cross_entropy_log_loss,
            'avg_cross_entropy_log_loss': avg_cross_entropy_log_loss,
            'labels': labels_,
            'onehot': onehot}

##############################################################################
def train(total_loss, global_step, parameters, flags):
    optimizer = tf.train.AdamOptimizer(learning_rate=flags['initial_learning_rate'])
    grads_and_vars = optimizer.compute_gradients(total_loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    return train_op


##############################################################################
def accuracy(predicts, labels, flags):
    # predicts - 32
    # labels = 32

    predicts = tf.Print(predicts, [predicts], message='Predicts: ',summarize = 32)
    labels = tf.Print(labels, [labels], message='Labels: ',summarize = 32)
    
    #predicts = tf.reshape(predicts, [-1])
    #labels = tf.reshape(labels, [-1])

    TP = [0] * flags['n_classes']
    FP = [0] * flags['n_classes']
    FN = [0] * flags['n_classes']
    TN = [0] * flags['n_classes']
    precision = [0] * flags['n_classes']
    recall = [0] * flags['n_classes']
    f1score = [0] * flags['n_classes']

    for iclass in range(flags['n_classes']):
        TP[iclass] = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(predicts, iclass), tf.equal(labels, iclass))))
        FP[iclass] = tf.reduce_sum(tf.to_float(tf.logical_and(tf.equal(predicts, iclass), tf.not_equal(labels, iclass))))
        FN[iclass] = tf.reduce_sum(tf.to_float(tf.logical_and(tf.not_equal(predicts, iclass), tf.equal(labels, iclass))))
        TN[iclass] = tf.reduce_sum(tf.to_float(tf.logical_and(tf.not_equal(predicts, iclass), tf.not_equal(labels, iclass))))

        precision[iclass] = TP[iclass] / tf.to_float(TP[iclass] + FP[iclass])
        recall[iclass] = TP[iclass] / tf.to_float(TP[iclass] + FN[iclass])
        f1score[iclass] = 2 * precision[iclass] * recall[iclass] / tf.to_float(precision[iclass] + recall[iclass])

    return {'TP': TP,
            'FP': FP,
            'FN': FN,
            'TN': TN,
            'precision': precision,
            'recall': recall,
            'f1score': f1score}
