import numpy as np
import tensorflow as tf
 
 
def vgg16(x, keep_prob, n_classes):
    def weight_variable(shape, name="weights"):
        initial = tf.truncated_normal(shape, dtype=tf.float32, stddev=0.1)
        return tf.Variable(initial, name=name)
 
    def bias_variable(shape, name="biases"):
        initial = tf.constant(0.1, dtype=tf.float32, shape=shape)
        return tf.Variable(initial, name=name)
 
    def conv2d(input, w):
        return tf.nn.conv2d(input, w, [1, 1, 1, 1], padding='SAME')
 
    def pool_max(input):
        return tf.nn.max_pool(input,
                              ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1],
                              padding='SAME',
                              name='pool1')
 
    def fc(input, w, b):
        return tf.matmul(input, w) + b
 
    # conv1
    with tf.name_scope('conv1_1') as scope:
        kernel = weight_variable([3, 3, 3, 64])
        biases = bias_variable([64])
        output_conv1_1 = tf.nn.relu(conv2d(x, kernel) + biases, name=scope)
 
    with tf.name_scope('conv1_2') as scope:
        kernel = weight_variable([3, 3, 64, 64])
        biases = bias_variable([64])
        output_conv1_2 = tf.nn.relu(conv2d(output_conv1_1, kernel) + biases, name=scope)
 
    pool1 = pool_max(output_conv1_2)
 
    # conv2
    with tf.name_scope('conv2_1') as scope:
        kernel = weight_variable([3, 3, 64, 128])
        biases = bias_variable([128])
        output_conv2_1 = tf.nn.relu(conv2d(pool1, kernel) + biases, name=scope)
 
    with tf.name_scope('conv2_2') as scope:
        kernel = weight_variable([3, 3, 128, 128])
        biases = bias_variable([128])
        output_conv2_2 = tf.nn.relu(conv2d(output_conv2_1, kernel) + biases, name=scope)
 
    pool2 = pool_max(output_conv2_2)
 
    # conv3
    with tf.name_scope('conv3_1') as scope:
        kernel = weight_variable([3, 3, 128, 256])
        biases = bias_variable([256])
        output_conv3_1 = tf.nn.relu(conv2d(pool2, kernel) + biases, name=scope)
 
    with tf.name_scope('conv3_2') as scope:
        kernel = weight_variable([3, 3, 256, 256])
        biases = bias_variable([256])
        output_conv3_2 = tf.nn.relu(conv2d(output_conv3_1, kernel) + biases, name=scope)
 
    with tf.name_scope('conv3_3') as scope:
        kernel = weight_variable([3, 3, 256, 256])
        biases = bias_variable([256])
        output_conv3_3 = tf.nn.relu(conv2d(output_conv3_2, kernel) + biases, name=scope)
 
    pool3 = pool_max(output_conv3_3)
 
    # conv4
    with tf.name_scope('conv4_1') as scope:
        kernel = weight_variable([3, 3, 256, 512])
        biases = bias_variable([512])
        output_conv4_1 = tf.nn.relu(conv2d(pool3, kernel) + biases, name=scope)
 
    with tf.name_scope('conv4_2') as scope:
        kernel = weight_variable([3, 3, 512, 512])
        biases = bias_variable([512])
        output_conv4_2 = tf.nn.relu(conv2d(output_conv4_1, kernel) + biases, name=scope)
 
    with tf.name_scope('conv4_3') as scope:
        kernel = weight_variable([3, 3, 512, 512])
        biases = bias_variable([512])
        output_conv4_3 = tf.nn.relu(conv2d(output_conv4_2, kernel) + biases, name=scope)
 
    pool4 = pool_max(output_conv4_3)
 
    # conv5
    with tf.name_scope('conv5_1') as scope:
        kernel = weight_variable([3, 3, 512, 512])
        biases = bias_variable([512])
        output_conv5_1 = tf.nn.relu(conv2d(pool4, kernel) + biases, name=scope)
 
    with tf.name_scope('conv5_2') as scope:
        kernel = weight_variable([3, 3, 512, 512])
        biases = bias_variable([512])
        output_conv5_2 = tf.nn.relu(conv2d(output_conv5_1, kernel) + biases, name=scope)
 
    with tf.name_scope('conv5_3') as scope:
        kernel = weight_variable([3, 3, 512, 512])
        biases = bias_variable([512])
        output_conv5_3 = tf.nn.relu(conv2d(output_conv5_2, kernel) + biases, name=scope)
 
    pool5 = pool_max(output_conv5_3)
 
    # fc6
    with tf.name_scope('fc6') as scope:
        shape = int(np.prod(pool5.get_shape()[1:]))
        kernel = weight_variable([shape, 4096])
        biases = bias_variable([4096])
        pool5_flat = tf.reshape(pool5, [-1, shape])
        output_fc6 = tf.nn.relu(fc(pool5_flat, kernel, biases), name=scope)
        dropout1 = tf.nn.dropout(output_fc6, keep_prob)
 
    # fc7
    with tf.name_scope('fc7') as scope:
        kernel = weight_variable([4096, 4096])
        biases = bias_variable([4096])
        output_fc7 = tf.nn.relu(fc(dropout1, kernel, biases), name=scope)
        dropout2 = tf.nn.dropout(output_fc7, keep_prob)
 
    # fc8
    with tf.name_scope('fc8') as scope:
        kernel = weight_variable([4096, n_classes])
        biases = bias_variable([n_classes])
        output_fc8 = fc(dropout2, kernel, biases)
#         output_fc8 = tf.nn.relu(fc(dropout2, kernel, biases), name=scope)
 
    # finaloutput = tf.nn.softmax(output_fc8, name="softmax")
 
    return output_fc8