
"""
preprocessing the datasets: affine distortion and elastic transformation
"""

import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
import numpy as np
from scipy.ndimage import interpolation
import input_data
import matplotlib.pyplot as plt

TRAIN_DIR = "MNIST_data/"
# 获取图片
mnist = input_data.read_data_sets(TRAIN_DIR)

# 获取训练集上图片
train = mnist.train
test = mnist.test

