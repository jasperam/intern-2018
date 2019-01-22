#!/usr/bin/python
# coding:utf-8
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected, l2_regularizer, l1_regularizer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import dataset
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

"""
定义DNN全连接层模型
"""
class Playground(object):
    def __init__(self, num_samples=500,
                 num_test_samples=200,
                 noise=0.05,
                 learning_rate=0.05,
                 layers=[3, 3],
                 act_func=tf.nn.relu,
                 act_output=tf.nn.tanh,
                 batch_size=10,
                 regularizer=None,
                 n_epochs=200
                 ):
        """

        :param num_samples: 训练集样本数
        :param num_test_samples: 测试集样本数
        :param noise: 噪声率
        :param learning_rate: 学习率
        :param layers: 隐藏层nodes list,[3,3]表示两个隐藏层，每层nodes个数为3
        :param act_func: 隐藏层激活函数
        :param act_output: 输出层激活函数
        :param batch_size: 小批量规模
        :param regularizer: 正则项，eg. l2_regularizer(scale=0.03)
        :param n_epochs: 遍历全数据集次数
        """
        self.g = tf.Graph()  # 全局计算图
        self.num_samples = num_samples
        self.num_test_samples = num_test_samples
        self.noise = noise
        self.learning_rate = learning_rate
        self.layers = layers
        self.act_fuc = act_func
        self.act_output = act_output
        self.batch_size = batch_size
        self.regularizer = regularizer
        self.n_epochs = n_epochs
        self.data = None  # 训练集X
        self.label = None  # 训练集Y
        self.test_data = None  # 测试集X
        self.test_label = None  # 测试集Y

    def set_graph(self):
        """
        绘制计算图，更改重要参数需要重置
        :return:
        """
        with self.g.as_default():
            X = tf.placeholder(dtype=tf.float32, shape=(None, 2), name="X")
            Y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="Y")
            assert len(self.layers), "隐藏层至少有一层"
            with tf.name_scope("dnn"):
                for layer in range(len(self.layers)):
                    if not layer:
                        hidden1 = fully_connected(X, self.layers[0], self.act_fuc, scope="hidden1",
                                                  weights_initializer=tf.random_uniform_initializer(minval=-0.5,
                                                                                                    maxval=0.5),
                                                  biases_initializer=tf.constant_initializer(0.1),
                                                  weights_regularizer=self.regularizer)
                    else:
                        exec('''hidden{}=fully_connected(hidden{}, self.layers[layer], 
                        self.act_fuc, scope="hidden{}", 
                        weights_initializer=tf.random_uniform_initializer(minval=-0.5, maxval=0.5),
                        biases_initializer=tf.constant_initializer(0.1),
                        weights_regularizer=self.regularizer)'''.format(layer+1, layer, layer+1))
                # output layers
                # name of y_ is mutable and dependent with activation function of output --> "act_output"
                exec('''y_=fully_connected(hidden{}, 1, 
                                        self.act_output, scope="output", 
                                        weights_initializer=tf.random_uniform_initializer(minval=-0.5, maxval=0.5),
                                        biases_initializer=tf.constant_initializer(0.1),
                                        weights_regularizer=self.regularizer)'''.format(len(self.layers)))
                # so use a copy of y_ and name it by ourselves for later indexing
                exec('''pred_y = tf.identity(y_, name="pred_y")''')
            with tf.name_scope("loss"):
                if self.regularizer:
                    reg_ws = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, 'dnn')
                    loss = tf.reduce_mean(
                        tf.square(self.g._nodes_by_name["dnn/pred_y"]._outputs[0] - Y), name="loss") + tf.reduce_sum(reg_ws)
                else:
                    loss = tf.reduce_mean(
                        tf.square(self.g._nodes_by_name["dnn/pred_y"]._outputs[0] - Y), name="loss")
            with tf.name_scope("train"):
                optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                training_op = optimizer.minimize(loss, name="training_op")
            with tf.name_scope("eval"):
                # 分类标准，pred_y>0.则分类为1；反之分类为0
                # 为输出拼接上一列全为0的向量，如果原输出为[0.98]，现在concat后为[0.,0.98]
                concat_preds = tf.concat(
                    [tf.zeros_like(self.g._nodes_by_name["dnn/pred_y"]._outputs[0], tf.float32),
                     self.g._nodes_by_name["dnn/pred_y"]._outputs[0]], axis=1)
                # 判断0.与原输出较大者的下标（0或1），并与真实的标签对比
                # 注：这里要求被对比的标签是1维的，所以用tf.squeeze降维
                #     且由于标签是-1和1，而下标是0或1，因此需要tf.clip_by_value将标签变为0~1
                correct = tf.nn.in_top_k(concat_preds,
                                         tf.cast(tf.clip_by_value(tf.squeeze(Y, [1]), 0., 1.), tf.int32), 1)
                accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
            init = tf.global_variables_initializer()

    def run_session(self, data_type):
        """
        执行运算会话，在此之前应设置好计算图
        :param data_type: 数据生成函数，请见dataset.py中的4类
        :return:
        """
        self.get_data(data_type)
        num = self.data.shape[0]
        # 每个epoch执行mini-batch的次数
        n_iters = num // self.batch_size
        self.loss_val = []
        self.acc_val = []
        self.loss_val_test = []
        self.acc_val_test = []
        with tf.Session(graph=self.g) as sess:
            sess.run(self.g._nodes_by_name["init"])
            print("DNN Model")
            for epoch in range(self.n_epochs):
                for itr in range(n_iters):
                    X_batch = self.data[itr * self.batch_size:(itr + 1) * self.batch_size, :]
                    Y_batch = self.label[itr * self.batch_size:(itr + 1) * self.batch_size, :]
                    sess.run(self.g._nodes_by_name["train/training_op"],
                             feed_dict={self.g._nodes_by_name["X"]._outputs[0]: X_batch,
                                        self.g._nodes_by_name["Y"]._outputs[0]: Y_batch})

                l, acc = sess.run([self.g._nodes_by_name["loss/loss"]._outputs[0],
                                   self.g._nodes_by_name["eval/accuracy"]._outputs[0]],
                                  feed_dict={self.g._nodes_by_name["X"]._outputs[0]: self.data,
                                    self.g._nodes_by_name["Y"]._outputs[0]: self.label})
                l_test, acc_test = sess.run([self.g._nodes_by_name["loss/loss"]._outputs[0],
                                   self.g._nodes_by_name["eval/accuracy"]._outputs[0]],
                                  feed_dict={self.g._nodes_by_name["X"]._outputs[0]: self.test_data,
                                             self.g._nodes_by_name["Y"]._outputs[0]: self.test_label})

                self.loss_val.append([epoch, l])
                self.acc_val.append([epoch, acc])
                self.loss_val_test.append([epoch, l_test])
                self.acc_val_test.append([epoch, acc_test])
                if not (epoch % 10):
                    print("epoch {}\ntraining_loss:{:.3f} test_loss:{:.3f}"
                          "\ntraining_accuracy:{:.2%} test_accuracy:{:.2%}".format(epoch, l, l_test, acc, acc_test))

    def plot_loss_and_accuracy(self):
        """
        分别画出训练集和测试集上的loss 和 accuracy rate图像
        :return:
        """
        loss_val = list(zip(*self.loss_val))
        acc_val = list(zip(*self.acc_val))
        loss_val_test = list(zip(*self.loss_val_test))
        acc_val_test = list(zip(*self.acc_val_test))
        plt.figure(1)
        plt.plot(np.array(loss_val[0]), np.array(loss_val[1]), color="red", label="training loss")
        plt.plot(np.array(loss_val_test[0]), np.array(loss_val_test[1]), color="blue", label="test loss")
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.title("loss on training set and test set")
        plt.legend()
        plt.show()

        plt.figure(2)
        ax = plt.gca()
        plt.plot(np.array(acc_val[0]), np.array(acc_val[1]) * 100, color="red", label="training accuracy")
        plt.plot(np.array(acc_val_test[0]), np.array(acc_val_test[1]) * 100, color="blue", label="test accuracy")
        plt.xlabel("epochs")
        plt.ylabel("accuracy rate")
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f%%"))
        plt.legend()
        plt.title("accuracy rate on training and test set")
        plt.show()

    def get_data(self, data_type):
        """
        获取训练集和测试集数据
        :param data_type:
        :return:
        """
        self.data, self.label = \
            dataset.get_samples(data_type, self.num_samples, self.noise, False)
        self.test_data, self.test_label = \
            dataset.get_samples(data_type, self.num_test_samples, self.noise, False)

    def reset_graph(self):
        """
        重置全局计算图
        :return:
        """
        self.g = tf.Graph()

    def reset_layers(self, new_layers):
        """
        重置隐藏层数
        :param new_layers: list，表示每个隐藏层的节点数
        :return:
        """
        self.layers = new_layers
        self.reset_graph()

    def reset_regularizer(self, new_regularizer):
        """
        重置正则项，常用L1,L2
        :param new_regularizer:
        :return:
        """
        self.regularizer = new_regularizer
        self.reset_graph()

    def reset_learning_rate(self, new_learning_rate):
        """
        重置学习率
        :param new_learning_rate:
        :return:
        """
        self.learning_rate = new_learning_rate

    def reset_num_samples(self, new_num_samples):
        """
        重设训练集样本量
        :param new_num_samples: int， 训练集样本量
        :return:
        """
        self.num_samples = new_num_samples

    def reset_num_test_samples(self, new_num_test_samples):
        """
        重设测试集样本量
        :param new_num_test_samples: int, 测试集样本量
        :return:
        """
        self.num_test_samples = new_num_test_samples

    def reset_act_func(self, new_act_func):
        """
        重设隐藏层激活函数
        :param new_act_func: function, 隐藏层激活函数
        :return:
        """
        self.act_fuc = new_act_func
        self.reset_graph()

    def reset_act_output(self, new_act_output):
        """
        重设输出激活函数
        :param new_act_output: function，输出激活函数
        :return:
        """
        self.act_output = new_act_output
        self.reset_graph()


if __name__ == "__main__":
    model = Playground()
    model.set_graph()
    model.reset_layers([4, 4])
    model.reset_regularizer(l2_regularizer(scale=0.03))
    model.set_graph()
    model.run_session(dataset.classify_two_gauss_data)
    model.plot_loss_and_accuracy()