# -*- coding: utf-8 -*-
# ! python3

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import input_data
from scipy.ndimage import interpolation


def moments(image):
    c0, c1 = np.mgrid[:image.shape[0], :image.shape[1]]
    totalImage = np.sum(image)
    m0 = np.sum(c0 * image) / totalImage
    m1 = np.sum(c1 * image) / totalImage
    m00 = np.sum((c0-m0)**2*image) / totalImage
    m11 = np.sum((c1-m0)**2*image) / totalImage
    m01 = np.sum((c0-m0)*(c1-m1)*image) / totalImage
    mu = np.array([m0, m1])
    covariance = np.array([[m00, m01], [m01, m11]])
    return mu, covariance


# affine transformation
def deskew(image):
    # c为计算得到实际重心
    image = image.reshape([28, 28])
    c, v = moments(image)
    alpha = v[0, 1] / v[0, 0]
    affine = np.array([[1, 0], [alpha, 1]])
    # center为图像原始重心，经过放射变换后应当变为实际重心c
    center = np.array(image.shape) / 2.0
    # offset为偏置向量b
    offset = c - np.dot(affine, center)
    image = interpolation.affine_transform(image, affine, offset=offset)
    image = image.reshape([-1,])
    return image


class MNIST_cnn:
    output_shape = 10
    input_height = 28
    input_width = 28
    n_channels = 1

    def __init__(self, display_step=10,
                 steps=200,
                 batch_size=50,
                 learning_rate=1e-3,
                 output_directory="mnist_CNN_log"):

        self.display_step = display_step
        self.steps = steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.output_directory = output_directory
        self.num_conv = 0
        self.graph = tf.Graph()
        self.saver, self.train_writer, self.test_writer, self.saver, self.merged = \
            None, None, None, None, None
        self.summary_flag = False

    def set_input(self):
        """
        reshape data's dims as 28*28 for each graph
        set data and label as placeholder
        :return: reshaped x and y_ as placeholder in self.graph
        """
        with self.graph.as_default():
            with tf.name_scope("input"):
                x = tf.placeholder(tf.float32,
                                   [None, self.input_height*self.input_width], name="input")
                y_ = tf.placeholder(tf.float32,
                                   [None, 10], name="label")
                x_reshaped = tf.reshape(tensor=x,
                                        shape=[-1, self.input_height, self.input_width, self.n_channels])
                tf.summary.image("input", x_reshaped, self.output_shape)
        return x, x_reshaped, y_

    def add_conv_layer(self, input,
                       filter_shape,
                       data_format="NHWC",
                       stride_height=1,
                       stride_width=1,
                       padding="VALID",
                       normalize=False,
                       activation_fuc=tf.nn.relu):
        """
        add convolution layer into the current model
        :param input: output or data from the previous layer or initial input
        :param filter_shape: convolution kernel' shape, list type, [height, width, in_chanels, out_chanels]
        :param data_format: input data format, str type; if "NHWC" [batch_num, height, width, channel],
        else "NCHW" [batch_num, channel, height, width]
        :param stride_height: stride in height axis, integer type
        :param stride_width: stride in width axis, integer type
        :param padding: str type: if "SAME", no padding; else "VALID" padding to keep output unshrink
        :param normalize: boolean type, do batch normalize or not
        :param activation_fuc: activation function; if not activated, use MNIST_cnn.fake_activation_func
        :return: _h, convolution output after activation before pooling
        """
        self.num_conv += 1
        with self.graph.as_default():
            with tf.name_scope("conv{}".format(self.num_conv)):
                with tf.name_scope("conv_kernel"):
                    _W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1))
                    with tf.name_scope("summaries"):
                        mean = tf.reduce_mean(_W)
                        tf.summary.scalar("mean", mean)
                        stddev = tf.sqrt(tf.reduce_mean(tf.square(_W - mean)))
                        tf.summary.scalar("stddev", stddev)
                        tf.summary.histogram("histogram", _W)
                with tf.name_scope("conv_preactivate"):
                    if normalize:
                        with tf.name_scope("batch_normalize"):
                            _act_conv = tf.nn.conv2d(input,
                                                _W,
                                                strides=[1, stride_height, stride_width, 1],
                                                padding=padding,
                                                data_format=data_format)
                            batch_mean, batch_var = tf.nn.moments(_act_conv,
                                                                  [0, 1, 2],
                                                                  keep_dims=True)
                            offset = tf.Variable(tf.zeros([filter_shape[-1]]))
                            scale = tf.Variable(tf.ones([filter_shape[-1]]))
                            _act = tf.nn.batch_normalization(_act_conv, mean=batch_mean,
                                                             variance=batch_var,
                                                             offset=offset,
                                                             scale=scale,
                                                             variance_epsilon=1e-4)

                    else:
                        with tf.name_scope("conv_bias"):
                            _b = tf.Variable(tf.constant(0.1, shape=[filter_shape[-1]]))
                            with tf.name_scope("summaries"):
                                mean = tf.reduce_mean(_b)
                                tf.summary.scalar("mean", mean)
                                stddev = tf.sqrt(tf.reduce_mean(tf.square(_b - mean)))
                                tf.summary.scalar("stddev", stddev)
                                tf.summary.histogram("histogram", _b)
                        _act = tf.nn.conv2d(input,
                                            _W,
                                            strides=[1, stride_height, stride_width, 1],
                                            padding=padding,
                                            data_format=data_format) + _b
                with tf.name_scope("conv_activation"):
                    _h = activation_fuc(_act)
                    tf.summary.histogram("pre_activation", _act)
                    tf.summary.histogram("activation", _h)
                    return _h

    def add_pool_layer(self, hidden, pool_type, ksize, strides, padding="VALID"):
        """
        add pooling layer to the current model
        :param hidden: convolution layer output after activation
        :param pool_type: str type, "max_pool" or "avg_pool"
        :param ksize: pooling kernel size, [1, h, w, 1]
        :param strides: strides size, [1, s1, s2, 1]
        :param padding: padding type, "SAME" or "VALID"
        :return: output after pooling
        """
        with self.graph.as_default():
            with tf.name_scope("conv{}".format(self.num_conv)):
                with tf.name_scope(pool_type):
                    if pool_type == "max_pool":
                        _pool = tf.nn.max_pool(hidden,
                                               ksize=ksize,
                                               strides=strides,
                                               padding=padding)
                    elif pool_type == "avg_pool":
                        _pool = tf.nn.avg_pool(hidden,
                                               ksize=ksize,
                                               strides=strides,
                                               padding=padding)
                    else:
                        print("Error: unsupported pooling type")
                        _pool = hidden
                    return _pool

    def set_image_output(self, pool, out_channel):
        with self.graph.as_default():
            with tf.name_scope("Image_output_conv{}".format(self.num_conv)):
                img = pool[0:1, :, :, 0:out_channel]
                tf.summary.image("Image_output_conv{}".format(self.num_conv),
                                 tf.transpose(img, perm=[3, 1, 2, 0]), out_channel)  # 交换第0维“N”和第3维“C”的维度

    def add_fully_connected_layer(self, input,
                                  num_nodes,
                                  activation_func=tf.nn.relu,
                                  keep_prob=0.8,
                                  normalize=False,
                                  scope="Fully_connected"):
        with self.graph.as_default():
            with tf.name_scope(scope):
                _flat = tf.reshape(input, [-1, np.prod(input.shape.as_list()[1:])])
                _Wfc = tf.Variable(tf.truncated_normal([np.prod(input.shape.as_list()[1:]), num_nodes], stddev=0.1))
                if normalize:
                    with tf.name_scope("batch_normalize"):
                        _flat_fc = tf.matmul(_flat, _Wfc)
                        batch_mean, batch_var = tf.nn.moments(_flat_fc, axes=[0, 1], keep_dims=True)
                        _hfc = tf.nn.batch_normalization(_flat_fc,
                                                         mean=batch_mean,
                                                         variance=batch_var,
                                                         offset=tf.zeros([1]),
                                                         scale=tf.ones([1]),
                                                         variance_epsilon=1e-4)
                        _hfc = activation_func(_hfc)

                else:
                    _bfc = tf.Variable(tf.constant(0.1, shape=[num_nodes]))
                    _hfc = activation_func(tf.matmul(_flat, _Wfc) + _bfc)
                _hfc_dropout = tf.nn.dropout(_hfc, keep_prob=keep_prob) if keep_prob < 1.0 else _hfc
                return _hfc_dropout

    def loss(self, labels, logits):
        with self.graph.as_default():
            with tf.name_scope("cross_entropy"):
                diff = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
                cross_entropy = tf.reduce_mean(diff)
            tf.summary.scalar("cross_entropy", cross_entropy)
            return cross_entropy

    def optimize(self, optimizer, cross_entropy):
        with self.graph.as_default():
            with tf.name_scope("train"):
                train = optimizer(self.learning_rate).minimize(cross_entropy)
                return train

    def accuracy(self, label, output):
        with self.graph.as_default():
            with tf.name_scope("accuracy"):
                pred = tf.equal(tf.argmax(label, 1), tf.argmax(output, 1))
                acc = tf.reduce_mean(tf.cast(pred, tf.float32))
            tf.summary.scalar("accuracy", acc)
            return acc

    def prepare_summaries(self):
        from pathlib import Path
        from shutil import rmtree
        p = Path(self.output_directory)
        if not p.exists():
            print("\nOutput directory does not exist - creating...")

        else:
            print("\nOutput directory already exist - overwritten...")
            rmtree(self.output_directory, ignore_errors=True)
        p.mkdir(parents=True)
        p1 = p / "train"
        p1.mkdir()
        p2 = p / "test"
        p2.mkdir()
        with self.graph.as_default():
            self.merged = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(str(p1))
            self.test_writer = tf.summary.FileWriter(str(p2))
            self.saver = tf.train.Saver()
            self.summary_flag = True

    def set_embedding(self, embedding_input, writer):
        with self.graph.as_default():
            with tf.name_scope("embedding"):
                embedding = tf.Variable(tf.zeros([1024,
                                                  embedding_input.get_shape().as_list()[1]]),
                                        name="test_embedding")
                assignment = embedding.assign(embedding_input)
                self.saver = tf.train.Saver()
                config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
                embedding_config = config.embeddings.add()
                embedding_config.tensor_name = embedding.name
                embedding_config.sprite.image_path = \
                    "C:/Master/Classes-arrangement/Year One/大岩实习/Learning/sprite_1024.png"
                embedding_config.metadata_path = \
                    "C:/Master/Classes-arrangement/Year One/大岩实习/Learning/labels_1024.tsv"
                embedding_config.sprite.single_image_dim.extend([28, 28])
                tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
                return assignment

    def run_session(self, mnist, loss, train, accuracy, x, y_, assignment=None):
        assert self.summary_flag, "fail to create summaries, stop session"
        with tf.Session(graph=self.graph) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(1, self.steps+1):
                batch_img, batch_label = mnist.train.next_batch(self.batch_size)
                test_batch = mnist.test.next_batch(self.batch_size)
                # 只用训练集上数据计算来改变所有参数，不在其他数据及上执行train的步骤
                sess.run(train, feed_dict={x: batch_img,
                                           y_: batch_label
                                           })
                if not i % self.display_step:
                    train_summary, train_loss, train_accuracy = \
                        sess.run([self.merged, loss, accuracy],
                                 feed_dict={x: batch_img,
                                            y_: batch_label
                                            })
                    self.train_writer.add_summary(train_summary, i)
                    print("step %d, training accuracy %g, training loss %g" % (i, train_accuracy, train_loss))
                    test_summary, test_loss, test_accuracy = sess.run([self.merged, loss, accuracy],
                                                                       feed_dict={x: test_batch[0],
                                                                                  y_: test_batch[1]
                                                                                  })
                    self.test_writer.add_summary(test_summary, i)
                    print("test accuracy %g, test loss %g" % (test_accuracy, test_loss))
                if i % 200 == 0 or i == self.steps:
                    print("-*-Adding run metadata for epoch {}-*-".format(i))
                    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                    run_metadata = tf.RunMetadata()
                    summary, _ = sess.run([self.merged, train],
                                          feed_dict={x: batch_img,
                                                     y_: batch_label},
                                          options=run_options,
                                          run_metadata=run_metadata)
                    if assignment is not None:
                        sess.run(assignment, feed_dict={x: mnist.test.images[:1024],
                                                        y_: mnist.test.labels[:1024]})
                    self.train_writer.add_run_metadata(run_metadata, "step%03d" % i)
                    self.train_writer.add_summary(summary, i)
                    print("\nSaving model at {} steps".format(i))
                    self.saver.save(sess,
                                    "{}/model_at_{}_steps.ckpt".format(self.output_directory, i),
                                    global_step=i)
            self.train_writer.add_graph(self.graph)
            self.train_writer.close()
            self.test_writer.close()

            train_accuracy = sess.run(accuracy,
                                      feed_dict={x: mnist.train.images,
                                                 y_: mnist.train.labels})
            print("\nFinal accuracy of train set: {:.4f}".format(train_accuracy))
            validation_accuracy = sess.run(accuracy,
                                           feed_dict={x: mnist.validation.images,
                                                      y_: mnist.validation.labels})
            print("Final accuracy of validation set: {:.4f}".format(validation_accuracy))
            test_accuracy = sess.run(accuracy,
                                     feed_dict={x: mnist.test.images,
                                                y_: mnist.test.labels})
            print("Final accuracy of test set: {:.4f}".format(test_accuracy))

    @staticmethod
    def fake_activation_func(*args):
        return args[0]


if __name__ == "__main__":
    TRAIN_DIR = "MNIST_data/"
    mnist = input_data.read_data_sets(TRAIN_DIR, one_hot=True)
    # -----------------------------------------------
    # use affine transformation to deskew graphs
    # -----------------------------------------------
    mnist.train.images = \
        np.apply_along_axis(deskew, 1, mnist.train.images)
    mnist.validation.images = \
        np.apply_along_axis(deskew, 1, mnist.validation.images)
    mnist.test.images = \
        np.apply_along_axis(deskew, 1, mnist.test.images)
    # -----------------------------------------------
    # set model: 2 convolution layers + 3 fully connected layers + softmax
    # use LeNet-5 model's parameters
    # -----------------------------------------------
    model = MNIST_cnn(steps=1000)
    init_data, data, label = model.set_input()
    # ---------------------------------------
    # Convolution Layer 1
    # kernel: 5*5
    # out channels: 6
    # strides: 1*1
    # act_func: ReLu
    hidden1 = model.add_conv_layer(input=data,
                                   filter_shape=[5, 5, 1, 6],
                                   stride_height=1,
                                   stride_width=1,
                                   normalize=True,
                                   activation_fuc=tf.nn.relu,
                                   padding="SAME")
    # ----------------------------------------
    # Max Pooling Layer 1
    # ksize: 2*2
    # strides: 2*2
    pool1 = model.add_pool_layer(hidden1,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 pool_type="max_pool",
                                 padding="SAME")
    # ----------------------------------------
    # save images output of the first pool layer
    # ----------------------------------------
    model.set_image_output(pool=pool1, out_channel=6)
    # ----------------------------------------
    # Convolution Layer 2
    # kernel: 5*5
    # out channels: 16
    # strides: 1*1
    # act_func: ReLu
    # padding="SAME"
    hidden2 = model.add_conv_layer(input=pool1,
                                   data_format="NHWC",
                                   padding="SAME",
                                   normalize=True,
                                   filter_shape=[5, 5, 6, 16],
                                   stride_height=1,
                                   stride_width=1,
                                   activation_fuc=tf.nn.relu)
    # ----------------------------------------
    # Max Pooling Layer 2
    # ksize: 2*2
    # strides: 2*2
    pool2 = model.add_pool_layer(hidden2,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 pool_type="max_pool",
                                 padding="SAME")
    # --------------------------------
    # Fully connected Layer 1
    # nodes: 120
    fc1 = model.add_fully_connected_layer(input=pool2,
                                          num_nodes=120,
                                          activation_func=tf.nn.relu,
                                          keep_prob=1.0,
                                          scope="fully-connected_1")
    # ---------------------------------
    # Fully Connected Layer 2
    # nodes: 84
    fc2 = model.add_fully_connected_layer(input=fc1,
                                          num_nodes=84,
                                          activation_func=tf.nn.relu,
                                          keep_prob=1.0,
                                          scope="fully-connected_2")
    # ---------------------------------
    # Fully Connected Layer 3 / output layer
    # nodes: 10
    logits = model.add_fully_connected_layer(input=fc2,
                                             num_nodes=10,
                                             activation_func=MNIST_cnn.fake_activation_func,
                                             keep_prob=1.0,
                                             scope="output_layer")
    # ---------------------------------
    # loss and accuracy & set summaries
    # ---------------------------------
    cross_entropy = model.loss(labels=label, logits=logits)
    train = model.optimize(optimizer=tf.train.AdamOptimizer,
                           cross_entropy=cross_entropy)
    accuracy = model.accuracy(label=label,
                              output=logits)
    model.prepare_summaries()
    # ---------------------------------
    # set embedding params
    # ---------------------------------
    assignment = model.set_embedding(embedding_input=fc2,
                                     writer=model.test_writer)
    model.run_session(mnist=mnist,
                      train=train,
                      loss=cross_entropy,
                      accuracy=accuracy,
                      x=init_data,
                      y_=label,
                      assignment=assignment)


