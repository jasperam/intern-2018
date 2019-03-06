# -*- coding: utf-8 -*-
# ! python3

"""
display dataset graphs
preprocessing the graphs: deskewing
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

# 看训练集的前500张图片
fig, ax = plt.subplots(nrows=2, ncols=5, sharex="all", sharey="all")
ax = ax.flatten()
ax[0].set_xticks([])
ax[0].set_yticks([])
plt.tight_layout()
for j in range(50):
    for i in range(10):
        img = train.images[i+10*j].reshape(28, 28)
        ax[i].imshow(img, cmap="Greys", interpolation="nearest")
    plt.pause(1)

# 看同一数字的不同样本
fig2, ax2 = plt.subplots(nrows=5, ncols=5, sharex="all", sharey="all")
ax2 = ax2.flatten()
ax2[0].set_xticks([])
ax2[0].set_yticks([])
plt.tight_layout()
# 存储同数字图片索引
num_idx_dict = {}
for num in range(10):
    set_idx = np.where(train.labels == num)[0]
    num_idx_dict.update({num: set_idx})
    num_set = train.images[set_idx]
    for k in range(25):
        img = num_set[k].reshape(28, 28)
        ax2[k].imshow(img, cmap="Greys", interpolation="nearest")
    plt.pause(1.5)


# make it simple, only shear transformation in the affine matrix
# 数据格式为np.ndrray-2D不是np.matrix
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
    c, v = moments(image)
    alpha = v[0, 1] / v[0, 0]
    affine = np.array([[1, 0], [alpha, 1]])
    # center为图像原始重心，经过放射变换后应当变为实际重心c
    center = np.array(image.shape) / 2.0
    # offset为偏置向量b
    offset = c - np.dot(affine, center)
    return interpolation.affine_transform(image, affine, offset=offset)


import random
examples = [random.choice(num_idx_dict[num]) for num in range(10)]
fig3, ax3 = plt.subplots(nrows=2, ncols=10, sharey="all", sharex="all", figsize=(11, 2))
ax3 = ax3.flatten()
ax3[0].set_xticks([])
ax3[0].set_yticks([])
for i, idx in enumerate(examples):
    img = train.images[idx].reshape(28, 28)
    ax3[i].imshow(img, interpolation="nearest")
    ax3[10+i].imshow(deskew(img), interpolation="nearest")
plt.show()
