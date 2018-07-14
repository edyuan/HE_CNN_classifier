import tensorflow as tf
from KS_lib.tf_op import layer

##############################################################################
def inference(images, hes, keep_prob, flags):

    images = tf.concat((images, hes),3)

    # 144x144
    with tf.variable_scope('down1'):
        with tf.variable_scope('conv1'):
            down1_conv1 = layer.down_conv_relu_same(images, [3, 3], 2*flags['size_input_patch'][2], 32)
        with tf.variable_scope('conv2'):
            down1_conv2 = layer.down_conv_relu_same(down1_conv1, [3, 3], 32, 32)
        with tf.variable_scope('pool'):
            down1_pool = layer.max_pool(down1_conv2, name='pool')

    # 72x72
    with tf.variable_scope('down2'):
        with tf.variable_scope('conv1'):
            down2_conv1 = layer.down_conv_relu_same(down1_pool, [3, 3], 32, 64)
        with tf.variable_scope('conv2'):
            down2_conv2 = layer.down_conv_relu_same(down2_conv1, [3, 3], 64, 64)
        with tf.variable_scope('pool'):
            down2_pool = layer.max_pool(down2_conv2, name='pool')

    # 36x36
    with tf.variable_scope('down3'):
        with tf.variable_scope('conv1'):
            down3_conv1 = layer.down_conv_relu_same(down2_pool, [3, 3], 64, 128)
        with tf.variable_scope('conv2'):
            down3_conv2 = layer.down_conv_relu_same(down3_conv1, [3, 3], 128, 128)
        with tf.variable_scope('pool'):
            down3_pool = layer.max_pool(down3_conv2, name='pool')

    # 18x18
    with tf.variable_scope('down4'):
        with tf.variable_scope('conv1'):
            down4_conv1 = layer.down_conv_relu_same(down3_pool, [3, 3], 128, 256)
        with tf.variable_scope('conv2'):
            down4_conv2 = layer.down_conv_relu_same(down4_conv1, [3, 3], 256, 256)
        with tf.variable_scope('pool'):
            down4_pool = layer.max_pool(down4_conv2, name='pool')

    # 9x9
    with tf.variable_scope('down5'):
        with tf.variable_scope('conv1'):
            down5_conv1 = layer.down_conv_relu_same(down4_pool, [3, 3], 256, 512)
        with tf.variable_scope('dropout1'):
            down5_drop1 = layer.dropout(down5_conv1,keep_prob)
        with tf.variable_scope('conv2'):
            down5_conv2 = layer.down_conv_relu_same(down5_drop1, [3, 3], 512, 512)
        with tf.variable_scope('dropout2'):
            down5_drop2 = layer.dropout(down5_conv2,keep_prob)
        with tf.variable_scope('tconv1'):
            down5_tconv = tf.image.resize_nearest_neighbor(down5_drop2, [18, 18])
        with tf.variable_scope('tconv1_'):
            down5_tconv = layer.down_conv_same(down5_tconv, [3, 3], 512, 256)
            # down5_tconv = layer.up_conv(down5_drop2, [2, 2], [2, 2], [18, 18], 512, 256, flags)

    # 18x18
    with tf.variable_scope('up1'):
        with tf.variable_scope('concat'):
            down4_conv2_down = layer.dropout(down4_conv2, keep_prob)
            up1_concat = tf.concat([down5_tconv, down4_conv2_down],3)
        with tf.variable_scope('conv1'):
            up1_conv1 = layer.down_conv_relu_same(up1_concat, [3, 3], 512, 256)
        with tf.variable_scope('conv2'):
            up1_conv2 = layer.down_conv_relu_same(up1_conv1, [3, 3], 256, 256)
        with tf.variable_scope('tconv1'):
            up1_tconv = tf.image.resize_nearest_neighbor(up1_conv2, [36, 36])
        with tf.variable_scope('tconv1_'):
            up1_tconv = layer.down_conv_same(up1_tconv, [3, 3], 256, 128)
            # up1_tconv = layer.up_conv(up1_conv2, [2, 2], [2, 2], [36, 36], 256, 128, flags)

    # 36x36
    with tf.variable_scope('up2'):
        with tf.variable_scope('concat'):
            down3_conv2_down = layer.dropout(down3_conv2, keep_prob)
            up2_concat = tf.concat([up1_tconv, down3_conv2_down],3)
        with tf.variable_scope('conv1'):
            up2_conv1 = layer.down_conv_relu_same(up2_concat, [3, 3], 256, 128)
        with tf.variable_scope('conv2'):
            up2_conv2 = layer.down_conv_relu_same(up2_conv1, [3, 3], 128, 128)
        with tf.variable_scope('tconv1'):
            up2_tconv = tf.image.resize_images(up2_conv2, [72, 72])
        with tf.variable_scope('tconv1_'):
            up2_tconv = layer.down_conv_same(up2_tconv, [3, 3], 128, 64)
            # up2_tconv = layer.up_conv(up2_conv2, [2, 2], [2, 2], [72, 72], 128, 64, flags)

    # 72x72
    with tf.variable_scope('up3'):
        with tf.variable_scope('concat'):
            down2_conv2_down = layer.dropout(down2_conv2, keep_prob)
            up3_concat = tf.concat([up2_tconv, down2_conv2_down], 3)
        with tf.variable_scope('conv1'):
            up3_conv1 = layer.down_conv_relu_same(up3_concat, [3, 3], 128, 64)
        with tf.variable_scope('conv2'):
            up3_conv2 = layer.down_conv_relu_same(up3_conv1, [3, 3], 64, 64)
        with tf.variable_scope('tconv1'):
            up3_tconv = tf.image.resize_nearest_neighbor(up3_conv2, [144, 144])
        with tf.variable_scope('tconv1_'):
            up3_tconv = layer.down_conv_same(up3_tconv, [3, 3], 64, 32)
            # up3_tconv = layer.up_conv(up3_conv2, [2, 2], [2, 2], [144, 144], 64, 32, flags)

    # 144x144
    with tf.variable_scope('up4'):
        with tf.variable_scope('concat'):
            down1_conv2_down = layer.dropout(down1_conv2, keep_prob)
            up4_concat = tf.concat([up3_tconv, down1_conv2_down], 3)
        with tf.variable_scope('conv1'):
            up4_conv1 = layer.down_conv_relu_same(up4_concat, [3, 3], 64, 32)
        with tf.variable_scope('conv2'):
            up4_conv2 = layer.down_conv_relu_same(up4_conv1, [3, 3], 32, 32)
        with tf.variable_scope('conv3'):
            up4_conv3 = layer.down_conv_same(up4_conv2, [3, 3], 32, 3)
        with tf.variable_scope('softmax'):
            softmax = tf.nn.softmax(up4_conv3)

    return softmax, \
           {'softmax':softmax}

##############################################################################
def loss(softmax, labels, weights, curr_epoch , flags):

    # number of postives and negatives
    pos_mask = tf.to_float(tf.equal(labels,1))
    neg_mask = tf.to_float(tf.equal(labels,0))
    bd_mask = tf.to_float(tf.equal(labels,2))

    n_pos = tf.to_float(tf.reduce_sum(pos_mask, [0, 1, 2, 3]))
    n_neg = tf.to_float(tf.reduce_sum(neg_mask, [0, 1, 2, 3]))
    n_bd = tf.to_float(tf.reduce_sum(bd_mask, [0, 1, 2, 3]))

    #######################################################
    # class imbalance weight

    max_val = tf.reduce_max(tf.stack([n_pos,n_neg,n_bd]))

    pos_weights = (pos_mask/n_pos)*max_val*1.0
    neg_weights = (neg_mask/n_neg)*max_val*1.0
    bd_weights = (bd_mask/n_bd)*max_val*1.0

    class_weights = pos_weights + neg_weights + bd_weights
    total_weights = class_weights + flags['alpha'] * weights

    # sigmoid loss
    labels = tf.squeeze(labels)
    labels = tf.cast(labels, tf.int64)
    # labels = tf.expand_dims(labels, dim=1)
    onehot = tf.one_hot(labels, depth=flags['n_classes'], on_value=1.0, off_value=0.0, axis=3)
    # onehot = tf.squeeze(onehot, dim=3)

    epsilon = 1e-6
    truncated_softmax = tf.clip_by_value(softmax, epsilon, 1.0 - epsilon)
    cross_entropy_log_loss = -tf.reduce_sum(onehot * tf.log(truncated_softmax), reduction_indices=[3], keep_dims=True)
    cross_entropy_log_loss = (total_weights) * cross_entropy_log_loss
    avg_cross_entropy_log_loss = tf.reduce_mean(cross_entropy_log_loss, reduction_indices=[0, 1, 2])

    return {'truncated_softmax': truncated_softmax,
            'cross_entropy_log_loss': cross_entropy_log_loss,
            'avg_cross_entropy_log_loss': avg_cross_entropy_log_loss,
            'labels': labels,
            'onehot': onehot,
            'pos_mask': pos_mask,
            'neg_mask': neg_mask,
            'bd_mask':bd_mask,
            'n_pos':n_pos,
            'n_neg':n_neg,
            'n_bd':n_bd,
            'class_weights':class_weights,
            'total_weights':total_weights}

##############################################################################
def train(total_loss, global_step, parameters, flags):
    optimizer = tf.train.AdamOptimizer(learning_rate=flags['initial_learning_rate'])
    grads_and_vars = optimizer.compute_gradients(total_loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    return train_op


##############################################################################
def accuracy(predicts, labels, flags):
    predicts = tf.reshape(predicts, [-1])
    labels = tf.reshape(labels, [-1])

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
