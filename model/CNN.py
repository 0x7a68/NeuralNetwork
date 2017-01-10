import tensorflow as tf

from util import io
import numpy as np
import sys


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


sess = tf.InteractiveSession()

W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

# conv layer_1
x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# conv layer_2
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# full connection
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# output layer: softmax
W_fc2 = weight_variable([1024, 8])
b_fc2 = bias_variable([8])

yprime = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
y_conv = tf.nn.softmax(yprime)
y_ = tf.placeholder(tf.float32, [None, 8])

# model training
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(yprime, y_))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_prediction = tf.equal(tf.arg_max(y_conv, 1), tf.arg_max(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

sess.run(tf.initialize_all_variables())

# read validation data
x_val, y_val = io.load_data('data/validation_image.txt', 'image')
x_val = np.reshape(x_val, (-1, 784))
y_val = np.reshape(y_val, (-1, 8))

# saver = tf.train.Saver([W_conv1, W_conv2, W_fc1, W_fc2, b_conv1, b_conv2, b_fc1, b_fc2])
# saver.restore(sess, 'data/cnn_weight_8000')
# sess.run([W_conv1, W_conv2, W_fc1, W_fc2, b_fc1, b_fc2, b_conv1, b_conv2])
#
# x_data, y_data = io.load_data('data/train_image.txt', 'image')
# for i in range(2001):
#     length = len(x_data)
#     x_batch = x_data[i * 50 % length: (i + 1) * 50 % length]
#     y_batch = y_data[i * 50 % length: (i + 1) * 50 % length]
#     x_batch = np.reshape(x_batch, (-1, 784))
#     y_batch = np.reshape(y_batch, (-1, 8))
#
#     if i % 100 == 0:
#         train_accuacy = accuracy.eval(feed_dict={x: x_batch, y_: y_batch, keep_prob: 1.0})
#         print("step %d, training accuracy %g" % (i, train_accuacy))
#
#         print("validation set accuracy %g" % (
#             accuracy.eval(feed_dict={x: x_val, y_: y_val, keep_prob: 1.0})))
#
#     train_step.run(feed_dict={x: x_batch, y_: y_batch, keep_prob: 0.5})
#
#
# print("train set accuracy %g" % (
#     accuracy.eval(feed_dict={x: x_data, y_: y_data, keep_prob: 1.0})))
#
# saver = tf.train.Saver([W_conv1, W_conv2, W_fc1, W_fc2, b_conv1, b_conv2, b_fc1, b_fc2])
# saver.save(sess, 'data/cnn_weight_10000')
#


saver = tf.train.Saver()

saver.restore(sess, 'data/cnn_weight')

sess.run([W_conv1, W_conv2, W_fc1, W_fc2, b_fc1, b_fc2, b_conv1, b_conv2])

TEST_IMAGE_TEXT_PATH = 'data/test_image'
OUTPUT_PATH = 'LetterCNN[14302010033].txt'

if __name__ == '__main__':
    if not len(sys.argv) == 3:
        print('Usage: CNN.py data_dir image_num')
        exit(0)

    io.image_to_txt(sys.argv[1], TEST_IMAGE_TEXT_PATH, False, sys.argv[2])
    input_data = io.load_input(TEST_IMAGE_TEXT_PATH)

    input_data = np.reshape(input_data, (-1, 784))

    result = sess.run(tf.argmax(y_conv, 1), feed_dict={x: input_data, keep_prob: 1.0})

    with open(OUTPUT_PATH, 'w') as f:
        for r in result:
            f.write(chr(r + ord('A')) + '\n')
