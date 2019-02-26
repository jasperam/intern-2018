# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.contrib.layers import fully_connected
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from dataset import * #可以直接使用dataset里面的函数

#circle data
data, label = get_samples(classify_circle_data, 50000, 0.1)

# define data in default graph
X = tf.placeholder(dtype=tf.float32, shape=(None, 2), name="input")
Y = tf.placeholder(dtype=tf.float32, shape=None, name="label")

