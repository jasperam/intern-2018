# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from dataset import *


# get data from dataset module, which is
data, label = get_samples(classify_circle_data, 500, 0.1)

# define data in default graph
X = tf.placeholder(dtype=tf.float32, shape=(None, 2), name="input")
Y = tf.placeholder(dtype=tf.float32, shape=None, name="label")

# try two hidden layers with 4 nodes each(3 layers in total)
n_hidden1 = 4
n_hidden2 = 4
n_output = 1
with tf.name_scope("dnn_circle"):
    hidden1 = fully_connected(X, n_hidden1, tf.nn.relu, scope="hidden1")
    hidden2 = fully_connected(hidden1, n_hidden2, tf.nn.relu, scope="hidden2")
    # only affine mapping, before activation function
    logits = fully_connected(hidden2, n_output, scope="output", activation_fn=None)

with tf.name_scope("loss"):
    pred_y = tf.nn.sigmoid(logits)
    xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    # loss = tf.reduce_mean(tf.square((pred_y - Y)))
learning_rate = 0.1

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

# 以准确率作为评价标准
with tf.name_scope("eval"):
    # correct = tf.nn.in_top_k(logits, Y, 1)  # 多分类问题
    concat_logits = tf.concat([tf.zeros_like(logits, dtype=tf.float32), logits], axis=1)
    correct = tf.nn.in_top_k(concat_logits, tf.cast(Y, tf.int32), 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
init = tf.global_variables_initializer()
n_epochs = 400
batch_size = 50
# 98.2%准确率
with tf.Session() as sess:
    init.run()
    num = data.shape[0]
    print("Model 1: circle data")
    for epoch in range(n_epochs):
        for iter in range(num // batch_size):
            X_batch = data[iter*batch_size: (iter+1)*batch_size, :]
            Y_batch = label[iter*batch_size: (iter+1)*batch_size]
            sess.run(training_op, feed_dict={X: X_batch, Y: Y_batch})

        if not ((epoch) % 50):
            preds, l, acc = sess.run([pred_y, loss, accuracy],
                                     feed_dict={X: data, Y: label})
            print("Prediction:\n{}, \nTrue label:\n{},"
                  " \nLoss:{:.3f}, \nAccuracy:{:.2%}".format(preds[::100], label[::100], l, acc))


# Model 2: Gauss data
g2 = tf.Graph()
with g2.as_default():
    data, label = get_samples(classify_two_gauss_data, 500, 0.1)
    # define data in default graph
    X = tf.placeholder(dtype=tf.float32, shape=(None, 2), name="input")
    Y = tf.placeholder(dtype=tf.float32, shape=None, name="label")

    # try two hidden layers with 4 nodes each(3 layers in total)
    n_hidden1 = 4
    n_hidden2 = 4
    n_output = 1
    with g2.name_scope("dnn_gauss"):
        hidden1 = fully_connected(X, n_hidden1, tf.nn.relu, scope="hidden1")
        hidden2 = fully_connected(hidden1, n_hidden2, tf.nn.relu, scope="hidden2")
        # only affine mapping, before activation function
        logits = fully_connected(hidden2, n_output, scope="output", activation_fn=None)
    with g2.name_scope("loss"):
        # pred_y = tf.nn.sigmoid(logits)
        # xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits)
        # loss = tf.reduce_mean(xentropy, name="loss")
        pred_y = tf.nn.relu(logits)
        loss = tf.reduce_mean(tf.square(pred_y-Y))
    learning_rate = 0.1
    with g2.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
    with g2.name_scope("eval"):
        concat_logits = tf.concat([tf.zeros_like(pred_y, tf.float32)+0.5, pred_y], axis=1)
        correct = tf.nn.in_top_k(concat_logits, tf.cast(Y, tf.int32), 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    init = tf.global_variables_initializer()
    batch_size = 50
    n_epochs = 400
    with tf.Session(graph=g2) as sess:
        init.run()
        print("Model 2: Gauss data")
        for epoch in range(n_epochs):
            num = data.shape[0]
            for iter in range(num // batch_size):
                X_batch = data[iter*batch_size:(iter+1)*batch_size, :]
                Y_batch = label[iter*batch_size:(iter+1)*batch_size]
                sess.run(training_op, feed_dict={X: X_batch, Y: Y_batch})
            if not ((epoch) % 50):
                preds, l, acc = sess.run([pred_y, loss, accuracy], feed_dict={X: data, Y: label})
                print("Prediction:\n{}, \nTrue label:\n{},"
                      " \nLoss:{:.3f}, \nAccuracy:{:.2%}".format(preds[::50], label[::50], l, acc))

# Model 3: xor data | harder than the previous two models
# smaller batch_size and little bit larger learning_rate
# > 97% accuracy
g3 = tf.Graph()
with g3.as_default():
    data, label = get_samples(classify_xor_data, 500, 0.05)
    # define data in default graph
    X = tf.placeholder(dtype=tf.float32, shape=(None, 2), name="input")
    Y = tf.placeholder(dtype=tf.float32, shape=None, name="label")

    # try two hidden layers with 4 nodes each(3 layers in total)
    n_hidden1 = 4
    n_hidden2 = 4
    n_output = 1

    # self-define leaky_relu
    # robuster than relu
    def leaky_relu(x, leak=0.2, name="leaky_relu"):
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * tf.abs(x)
    with g3.name_scope("dnn_xor"):
        hidden1 = fully_connected(X, n_hidden1, leaky_relu, scope="hidden1")
        hidden2 = fully_connected(hidden1, n_hidden2, leaky_relu, scope="hidden2")
        # only affine mapping, before activation function
        logits = fully_connected(hidden2, n_output, scope="output", activation_fn=None)
    with g3.name_scope("loss"):
        pred_y = tf.nn.sigmoid(logits)
        xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits)
        loss = tf.reduce_mean(xentropy, name="loss")
        # pred_y = tf.nn.relu(logits)
        # loss = tf.reduce_mean(tf.square(pred_y-Y))
    learning_rate = tf.placeholder(tf.float32)
    with g3.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
    with g3.name_scope("eval"):
        # concat_logits = tf.concat([tf.zeros_like(logits, tf.float32), logits], axis=1)
        concat_logits = tf.concat([tf.zeros_like(pred_y, tf.float32)+0.5, pred_y], axis=1)
        correct = tf.nn.in_top_k(concat_logits, tf.cast(Y, tf.int32), 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    init = tf.global_variables_initializer()
    n_epochs = 400
    batch_size = 5
    with tf.Session(graph=g3) as sess:
        init.run()
        print("Model 3: xor data")
        for epoch in range(n_epochs):
            num = data.shape[0]
            if epoch < 200:
                lr = 0.12
            else:
                lr = 0.1
            for iter in range(num // batch_size):
                X_batch = data[iter*batch_size:(iter+1)*batch_size, :]
                Y_batch = label[iter*batch_size:(iter+1)*batch_size]
                sess.run(training_op, feed_dict={X: X_batch, Y: Y_batch, learning_rate: lr})

            if not (epoch % 30):
                preds, l, acc = sess.run([pred_y, loss, accuracy], feed_dict={X: data, Y: label})
                print("Prediction:\n{}, \nTrue label:\n{},"
                      " \nLoss:{:.3f}, \nAccuracy:{:.2%}".format(preds[::100], label[::100], l, acc))

# Model 4: Spiral data
# hardest, use larger scale model to train
g4 = tf.Graph()
with g4.as_default():
    data, label = get_samples(classify_spiral_data, 500, 0.05)
    # define data in default graph
    X = tf.placeholder(dtype=tf.float32, shape=(None, 2), name="input")
    Y = tf.placeholder(dtype=tf.float32, shape=None, name="label")

    # try five hidden layers(6 layers in total)
    n_hidden1 = 8
    n_hidden2 = 8
    n_hidden3 = 7
    n_hidden4 = 6
    n_hidden5 = 5
    n_output = 1

    # self-define leaky_relu
    # robuster than relu
    def leaky_relu(x, leak=0.2, name="leaky_relu"):
        with tf.variable_scope(name):
            f1 = 0.5 * (1 + leak)
            f2 = 0.5 * (1 - leak)
            return f1 * x + f2 * tf.abs(x)
    with g4.name_scope("dnn_spiral"):
        hidden1 = fully_connected(X, n_hidden1, leaky_relu, scope="hidden1")
        hidden2 = fully_connected(hidden1, n_hidden2, leaky_relu, scope="hidden2")
        hidden3 = fully_connected(hidden2, n_hidden3, leaky_relu, scope="hidden3")
        hidden4 = fully_connected(hidden3, n_hidden4, leaky_relu, scope="hidden4")
        hidden5 = fully_connected(hidden4, n_hidden5, leaky_relu, scope="hidden5")
        # only affine mapping, before activation function
        logits = fully_connected(hidden5, n_output, scope="output", activation_fn=None)
    with g4.name_scope("loss"):
        pred_y = leaky_relu(logits)
        # xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits)
        # loss = tf.reduce_mean(xentropy, name="loss")
        loss = tf.reduce_mean(tf.square(pred_y-Y))
    learning_rate = tf.placeholder(tf.float32)
    with g4.name_scope("train"):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        training_op = optimizer.minimize(loss)
    with g4.name_scope("eval"):
        concat_logits = tf.concat([tf.zeros_like(logits, tf.float32) + 0.5, logits], axis=1)
        correct = tf.nn.in_top_k(concat_logits, tf.cast(Y, tf.int32), 1)
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    init = tf.global_variables_initializer()
    n_epochs = 1000
    batch_size = 15
    with tf.Session(graph=g4) as sess:
        init.run()
        print("Model 4: spiral data")
        # to_shuffle = True
        for epoch in range(n_epochs):
            num = data.shape[0]
            if epoch < 400:
                lr = 0.12

            # elif to_shuffle:
            #     np.random.shuffle(data[0])
            #     np.random.shuffle(data[1])
            #     np.random.shuffle(label)
            #     to_shuffle = False
            #     lr=0.09
            #     batch_size = 20
            else:
                lr = 0.1
            for iter in range(num // batch_size):
                X_batch = data[iter*batch_size:(iter+1)*batch_size, :]
                Y_batch = label[iter*batch_size:(iter+1)*batch_size]
                sess.run(training_op, feed_dict={X: X_batch, Y: Y_batch, learning_rate: lr})

            if not (epoch % 50):
                # sess.run(training_op, feed_dict={X: data, Y: label})
                preds, l, acc = sess.run([pred_y, loss, accuracy], feed_dict={X: data, Y: label})
                print("Prediction:\n{}, \nTrue label:\n{},"
                      " \nLoss:{:3.f}, \nAccuracy:{:.2%}".format(preds[::100], label[::100], l, acc))

