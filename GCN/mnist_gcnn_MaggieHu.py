# -*- coding: utf-8 -*-
# ! python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import math
import tensorflow_datasets as tfds


class MNIST_cnn:
    output_shape = 10
    input_height = 28
    input_width = 28
    U = 4

    def __init__(self, display_step=10,
                 steps=200,
                 batch_size=128,
                 learning_rate=1e-3,
                 output_directory="mnist_GCNN_log"):

        self.display_step = display_step
        self.steps = steps
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.output_directory = output_directory
        self.num_conv = 0
        self.graph = tf.Graph()
        self.saver, self.train_writer, self.test_writer, self.merged = \
            None, None, None, None
        self.summary_flag = False
        self.gabor_filter_banks = dict()

    def feed_data(self, dataset_train, dataset_test, test_batch):
        with self.graph.as_default():
            iter_train = dataset_train.repeat().batch(self.batch_size).make_one_shot_iterator()
            train = iter_train.get_next()
            x_train, y_train = train["image"], train["label"]
            iter_test = dataset_test.repeat().batch(min(test_batch, 10000)).make_one_shot_iterator()
            test = iter_test.get_next()
            x_test, y_test = test["image"], test["label"]
            return x_train, y_train, x_test, y_test

    def set_input(self):
        """
        reshape data's dims as 1*28*28*1 for each graph
        set data and label as placeholder
        :return: reshaped x and y_ as placeholder in self.graph
        """
        with self.graph.as_default():
            with tf.name_scope("input"):
                x_ = tf.placeholder(tf.float32,
                                   [None, self.input_height, self.input_width, 1], name="input")
                y_ = tf.placeholder(tf.int64,
                                    [None, ], name="label")
                x_reshaped = tf.reshape(tensor=x_,
                                        shape=[-1, 1, self.input_height, self.input_width, 1])
                # extend the depth(second axis) to be four
                x_reshaped = tf.tile(x_reshaped, [1, self.U, 1, 1, 1])
                # tf.summary.image("input", x_reshaped, self.output_shape)
        return x_, x_reshaped, y_

    def get_gabor_banks(self, u, v, w, h, sigma=math.pi*2):
        kmax = math.pi / 2
        f = math.sqrt(2)
        sqsigma = sigma ** 2
        postmean = math.exp(-sqsigma / 2)
        if h != 1:
            gfilter = np.zeros([u, h, w], dtype=np.float32)
            for i in range(u):
                theta = i / u * math.pi
                k = kmax / f ** (v - 1)
                xymax = -1e30
                xymin = 1e30
                for y in range(h):
                    for x in range(w):
                        y_ = y + 1 - ((h + 1) / 2)
                        x_ = x + 1 - ((w + 1) / 2)
                        tmp1 = math.exp(- (k ** 2 * (x_ ** 2 + y_ ** 2) / (2 * sqsigma)))
                        tmp2 = math.cos(k * math.cos(theta) * x_ + k * math.sin(theta) * y_) - postmean
                        gfilter[i][y][x] = k ** 2 *tmp1 *tmp2 / sqsigma
                        xymax = max(xymax, gfilter[i][y][x])
                        xymin = max(xymin, gfilter[i][y][x])
                gfilter[i] = (gfilter[i] - xymin) / (xymax - xymin)
        else:
            gfilter = np.ones([u, h, w], dtype=np.float32)
        return gfilter

    def add_conv_layer(self, input,
                       filter_shape,
                       v,
                       data_format="NDHWC",
                       stride_height=1,
                       stride_width=1,
                       stride_depth=1,
                       padding="VALID",
                       ):
        """
        add convolution layer into the current model
        :param input: output or data from the previous layer or initial input
        :param v: int，scale paramter of gabor filter in this layer
        :param filter_shape: convolution kernel' shape, list type, [depth, height, width, in_chanels, out_chanels]
        :param data_format: input data format, str type; if "NDHWC" [batch_num, depth, height, width, channel],
        else "NDCHW" [batch_num, depth, channel, height, width]
        :param stride_height: stride in height axis, integer type
        :param stride_width: stride in width axis, integer type
        :param padding: str type: if "SAME", no padding; else "VALID" padding to keep output unshrink
        :return: _h, convolution output after activation before pooling
        """
        self.num_conv += 1
        with self.graph.as_default():
            with tf.name_scope("conv{}".format(self.num_conv)):
                with tf.name_scope("conv_kernel"):
                    w_shape = filter_shape.copy()
                    w_shape.pop(0)
                    w_shape.reverse()
                    # shape: [out_channels, in_channels, height, width]
                    _W = tf.Variable(tf.truncated_normal(w_shape, stddev=0.1))
                    # 得到对应gabor filters, ahspe: [U, height, width]
                    gabor = self.gabor_filter_banks.get((self.U, v), None)
                    if not gabor:
                        gabor = self.get_gabor_banks(u=self.U,
                                                     v=v, h=filter_shape[1],
                                                     w=filter_shape[2])
                        self.gabor_filter_banks.update({(self.U, v): gabor})
                    gabor = tf.constant(gabor)
                    gabor_W = []
                    for o in range(w_shape[0]):
                        inner = []
                        for g in range(self.U):
                            inner.append(_W[o] * gabor[g])
                        inner = tf.stack(inner, axis=0)
                        gabor_W.append(inner)
                    # shape: [out_channels, U, in_channels, height, width]
                    gabor_W = tf.stack(gabor_W, axis=0, name="filter")
                    # shape: [out_channels, U, in_channels, depth, height, width]
                    gabor_W_ = tf.expand_dims(gabor_W, axis=3)
                    # 在depth维度上扩张每层2D filter
                    # gabor_W_ = tf.tile(gabor_W_, [1, 1, 1, self.U, 1, 1])
                    # 交换维度 shape: [U, depth, height, width, in_channel, out_channels]
                    gabor_W_ = tf.transpose(gabor_W_, perm=[1, 3, 4, 5, 2, 0])

                    # 需要多次卷积并将卷积结果进行堆叠组合
                    maps = []
                    for i in range(self.U):
                        conv_ = tf.nn.conv3d(input,
                                            filter=gabor_W_[i],
                                            strides=[1, stride_depth, stride_height, stride_width, 1],
                                            padding=padding,
                                            data_format=data_format,
                                            )
                        # 在depth维度上求和，但保留该维度
                        conv_ = tf.reduce_sum(conv_, axis=1, keepdims=True)
                        maps.append(conv_)
                    # 由于depth维度上的卷积核长度为depth=self.U，输出在depth维度上只有1
                    conv3 = tf.concat(maps, axis=1)
        return conv3

    def add_batch_normalization_layer(self, input):
        with self.graph.as_default():
            with tf.name_scope("batch_normalized"):
                # 保留depth维度上的独立性
                batch_mean, batch_var = tf.nn.moments(input,
                                                      [0, 2, 3],
                                                      keep_dims=True)
                offset = tf.Variable(tf.zeros(batch_mean.shape.as_list()))
                scale = tf.Variable(tf.ones(batch_mean.shape.as_list()))
                bn = tf.nn.batch_normalization(input,
                                               mean=batch_mean,
                                               variance=batch_var,
                                               offset=offset,
                                               scale=scale,
                                               variance_epsilon=1e-4)
        return bn

    def add_pool_layer(self, hidden, pool_type, ksize, strides, activation=tf.nn.relu, padding="VALID"):
        """
        add pooling layer to the current model
        :param hidden: convolution layer output before activation after batch_normalization
        :param pool_type: str type, "max_pool" or "avg_pool"
        :param ksize: pooling kernel size, [1, d, h, w, 1]
        :param strides: strides size, [s1, s2, s3]
        :param padding: padding type, "SAME" or "VALID"
        :param activation: 激活函数
        :return: output after pooling
        """
        with self.graph.as_default():
            with tf.name_scope(pool_type):
                if pool_type == "max_pool":
                    _pool = tf.nn.max_pool3d(hidden,
                                             ksize=ksize,
                                             strides=[1, *strides, 1],
                                             padding=padding)
                elif pool_type == "avg_pool":
                    _pool = tf.nn.avg_pool3d(hidden,
                                             ksize=ksize,
                                             strides=[1, *strides, 1],
                                             padding=padding)
                else:
                    # print("Error: unsupported pooling type")
                    _pool = hidden
                _pool = activation(_pool)
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
                len_ = np.prod(input.shape.as_list()[1:])
                _flat = tf.reshape(input, [-1, len_])
                _Wfc = tf.Variable(tf.truncated_normal([len_, num_nodes], stddev=0.1))
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
                labels = tf.one_hot(labels,
                                    depth=self.output_shape,
                                    dtype=tf.float32)
                diff = tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=logits)
                cross_entropy = tf.reduce_mean(diff)
            tf.summary.scalar("cross_entropy", cross_entropy)
            return cross_entropy

    def optimize(self, optimizer, cross_entropy):
        with self.graph.as_default():
            with tf.name_scope("train"):
                self.learning_rate = tf.placeholder(dtype=tf.float32)
                train = optimizer(self.learning_rate).minimize(cross_entropy)
                return self.learning_rate, train

    def accuracy(self, label, output):
        with self.graph.as_default():
            with tf.name_scope("accuracy"):
                pred = tf.equal(label, tf.argmax(output, 1))
                acc = tf.reduce_mean(tf.cast(pred, tf.float32))
            tf.summary.scalar("accuracy", acc)
            return acc

    def prepare_summaries(self, rm=True):

        from pathlib import Path
        from shutil import rmtree
        p = Path(self.output_directory)
        p1 = p / "train"
        p2 = p / "test"
        if rm:
            if not p.exists():
                print("\nOutput directory does not exist - creating...")

            else:
                print("\nOutput directory already exist - overwritten...")
                rmtree(self.output_directory, ignore_errors=True)
            p.mkdir(parents=True)
            p1.mkdir()
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
                    "C:/Master/Classes-arrangement/Year One/Jasper/Learning/sprite_1024.png"
                embedding_config.metadata_path = \
                    "C:/Master/Classes-arrangement/Year One/Jasper/Learning/labels_1024.tsv"
                embedding_config.sprite.single_image_dim.extend([28, 28])
                tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)
                return assignment

    def run_session(self, loss, lr_, train, accuracy, x_, y_, restore=False):
        assert self.summary_flag, "fail to create summaries, stop session"
        x_train, y_train, x_test, y_test = self.feed_data(dataset_train=mnist_train,
                                                          dataset_test=mnist_test,
                                                          test_batch=1000)
        with self.graph.as_default():
            with tf.Session() as sess:
                # sess.run(tf.global_variables_initializer())
                if restore:
                    self.saver.restore(sess,
                                       tf.train.latest_checkpoint("./" + self.output_directory))
                else:
                    sess.run(tf.global_variables_initializer())
                for i in range(1, self.steps+1):
                    # 每优化200次衰减为原来的0.9
                    lr = 0.001 * 0.9 ** int(i / 200)
                    batch_img, batch_label = sess.run([x_train, y_train])
                    # 只用训练集上数据计算来改变所有参数，不在其他数据及上执行train的步骤
                    sess.run(train, feed_dict={x_: batch_img,
                                               y_: batch_label,
                                               lr_: lr
                                               })
                    if not i % self.display_step:
                        train_summary, train_loss, train_accuracy = \
                            sess.run([self.merged, loss, accuracy],
                                     feed_dict={x_: batch_img,
                                                y_: batch_label
                                                })
                        self.train_writer.add_summary(train_summary, i)
                        print("step %d, training accuracy %g, training loss %g" % (i, train_accuracy, train_loss))

                    if i % 100 == 0 or i == self.steps:
                        batch_img_test, batch_label_test = sess.run([x_test, y_test])
                        acc_test = []
                        los_test = []
                        for _ in range(10):
                            loss_test, accuracy_test = \
                                sess.run([loss, accuracy],
                                         feed_dict={x_: batch_img_test,
                                         y_: batch_label_test},)
                            acc_test.append(accuracy_test)
                            los_test.append(loss_test)
                        accuracy_test = float(np.mean(acc_test))
                        loss_test = float(np.mean(los_test))
                        summary_test = sess.run(self.merged,
                                                feed_dict={x_: batch_img_test,
                                                           y_: batch_label_test})
                        print("test accuracy %g, test loss %g" % (accuracy_test, loss_test))
                        # self.train_writer.add_run_metadata(run_metadata, "step%03d" % i)
                        self.test_writer.add_summary(summary_test, i)
                    if i % 500 == 0 or i == self.steps:
                        print("\nSaving model at {} steps".format(i))
                        self.saver.save(sess,
                                        "{}/model_at_{}_steps.ckpt".format(self.output_directory, i),
                                        global_step=i)

                self.train_writer.add_graph(self.graph)
                self.train_writer.close()
                self.test_writer.close()
                # loss and accuracy in whole train data and test data


    @staticmethod
    def fake_activation_func(*args):
        return args[0]

    def print_W_initial_values(self, _W, out_channel):
        sess = tf.InteractiveSession(graph=self.graph)
        init = tf.initializers.global_variables()
        init.run()
        ww = tf.identity(_W)
        ww = tf.transpose(ww, [3, 2, 0, 1])
        print(f"conv layer weights: filter/out_channel {out_channel}")
        print(ww.eval()[out_channel, :, :, :])
        sess.close()

    def print_bias_initial_values(self, bias):
        sess = tf.InteractiveSession(graph=self.graph)
        init = tf.initializers.global_variables()
        init.run()
        print(f"conv layer bias:")
        print(bias.eval())
        sess.close()

    def print_variable_num(self, scope=None):
        with self.graph.as_default():
            print("trainable variables' num in scope {}: {}".format(scope,
                                            len(tf.trainable_variables(scope=scope))))
            print(f"they are: \n {[var.name for var in tf.trainable_variables(scope=scope)]}")
            print(f"global variables' num: {len(tf.global_variables())}")
            print(f"they are: \n {[var.name for var in tf.all_variables()]}")

    @staticmethod
    def print_variable_size(var):
        print(f"this variable has size: {np.product(var.shape.as_list())}")


if __name__ == "__main__":
    TRAIN_DIR = r"~\MNIST_data"

    # -----------------------------------------------
    # import dataset of Mnist: train, validation and test sets
    # -----------------------------------------------
    mnist_train = tfds.load(name="mnist", data_dir=TRAIN_DIR, split=tfds.Split.TRAIN, download=True)
    mnist_test = tfds.load(name="mnist", data_dir=TRAIN_DIR, split=tfds.Split.TEST, download=True)

    # -----------------------------------------------
    # set model: 2 convolution layers + 3 fully connected layers + softmax
    # use LeNet-5 model's parameters
    # -----------------------------------------------
    model = MNIST_cnn(steps=2000,
                      output_directory="mnist_GCNN_log2",
                      display_step=20)
    print("Initialize GCNN model")
    x_, data, y_ = model.set_input()
    print("%Input layer: [?, 4, 28, 28, 1]")
    # ---------------------------------------
    # Convolution Layer 1
    # in channels: 1
    # kernel: 4*3*3
    # out channels: 20
    # strides: 1*1*1
    hidden1 = model.add_conv_layer(input=data,
                                   v=1,
                                   filter_shape=[4, 3, 3, 1, 20],
                                   stride_height=1,
                                   stride_width=1,
                                   stride_depth=1,
                                   padding="SAME")
    print("%Conv3d layer 1: [?, 4, 28, 28, 20]")
    # ----------------------------------------
    # batch normalization layer
    bn1 = model.add_batch_normalization_layer(hidden1)
    print("%BN layer 1")
    # ----------------------------------------
    # Max Pooling Layer 1
    # ksize: 1*2*2
    # strides: 1*2*2
    # activation: relu
    pool1 = model.add_pool_layer(bn1,
                                 ksize=[1, 1, 2, 2, 1],
                                 strides=[1, 2, 2],
                                 pool_type="max_pool",
                                 padding="VALID")
    print("%Max_pooling layer 1: [?, 4, 14, 14, 20]")
    # ---------------------------------------
    # Convolution Layer 2
    # in channels: 20
    # kernel: 4*3*3
    # out channels: 40
    # strides: 1*1*1
    hidden2 = model.add_conv_layer(input=pool1,
                                   v=2,
                                   filter_shape=[4, 3, 3, 20, 40],
                                   stride_height=1,
                                   stride_width=1,
                                   stride_depth=1,
                                   padding="SAME")
    print("%Conv3d layer 2: [?, 4, 14, 14, 40]")
    # ----------------------------------------
    # batch normalization layer 2
    bn2 = model.add_batch_normalization_layer(hidden2)
    print("%BN layer 2")
    # ----------------------------------------
    # Max Pooling Layer 2
    # ksize: 1*2*2
    # strides: 1*2*2
    # activation: relu
    pool2 = model.add_pool_layer(bn2,
                                 ksize=[1, 1, 2, 2, 1],
                                 strides=[1, 2, 2],
                                 pool_type="max_pool",
                                 padding="VALID")
    print("%Max_pooling layer 2: [?, 4, 7, 7, 40]")
    # ---------------------------------------
    # Convolution Layer 3
    # in channels: 40
    # kernel: 4*3*3
    # out channels: 80
    # strides: 1*1*1
    hidden3 = model.add_conv_layer(input=pool2,
                                   v=3,
                                   filter_shape=[4, 3, 3, 40, 80],
                                   stride_height=1,
                                   stride_width=1,
                                   stride_depth=1,
                                   padding="SAME")
    print("%Conv3d layer 3: [?, 4, 7, 7, 80]")
    # ----------------------------------------
    # batch normalization layer 3
    bn3 = model.add_batch_normalization_layer(hidden3)
    print("%BN layer 3")
    # ----------------------------------------
    # Max Pooling Layer 3
    # ksize: 1*2*2
    # strides: 1*2*2
    # activation: relu
    pool3 = model.add_pool_layer(bn3,
                                 ksize=[1, 1, 2, 2, 1],
                                 strides=[1, 2, 2],
                                 pool_type="max_pool",
                                 padding="VALID")
    print("%Max_pooling layer 3: [?, 4, 3, 3, 80]")
    # ---------------------------------------
    # Convolution Layer 4
    # in channels: 80
    # kernel: 4*3*3
    # out channels: 160
    # strides: 1*1*1
    hidden4 = model.add_conv_layer(input=pool3,
                                   v=4,
                                   filter_shape=[4, 3, 3, 80, 160],
                                   stride_height=1,
                                   stride_width=1,
                                   stride_depth=1,
                                   padding="VALID")
    print("%Conv3d layer 4: [?, 4, 1, 1, 160]")
    # ----------------------------------------
    # batch normalization layer 4
    bn4 = model.add_batch_normalization_layer(hidden4)
    print("%BN layer 4")
    # ----------------------------------------
    # Max Pooling Layer 4
    # no pooling
    # activation: relu
    pool4 = model.add_pool_layer(bn4,
                                 ksize=[],
                                 strides=[],
                                 pool_type="None")
    print("%No pooling: only ReLu activation")
    # fully_connected之前在depth为维度上max
    pool4 = tf.reduce_max(pool4, axis=1)
    print("%Max in depth-axis: [?, 1, 1, 1, 160]")
    # --------------------------------
    # Fully connected Layer 1
    # nodes: 1024
    # dropout: 0.8
    fc1 = model.add_fully_connected_layer(input=pool4,
                                          num_nodes=1024,
                                          activation_func=tf.nn.relu,
                                          keep_prob=0.8,
                                          scope="fully-connected_1")
    print("%Fully_connected layer 1: [?, 1024]")
    # ---------------------------------
    # Fully Connected Layer 3 / output layer
    # nodes: 10
    logits = model.add_fully_connected_layer(input=fc1,
                                             num_nodes=10,
                                             activation_func=MNIST_cnn.fake_activation_func,
                                             keep_prob=1.0,
                                             scope="output_layer")
    print("%Fully_cinnected layer 2/Output layer: [?, 10]")
    # ---------------------------------
    # loss and accuracy & set summaries
    # ---------------------------------
    cross_entropy = model.loss(labels=y_, logits=logits)
    lr, train = model.optimize(optimizer=tf.train.AdamOptimizer,
                               cross_entropy=cross_entropy)
    accuracy = model.accuracy(label=y_,
                              output=logits)
    model.prepare_summaries(rm=True)
    import time
    start = time.time()
    model.run_session(lr_=lr,
                      train=train,
                      loss=cross_entropy,
                      accuracy=accuracy,
                      x_=x_,
                      y_=y_,
                      restore=False
                      )
    end = time.time()
    print("Run time: %g s" % (end - start))



