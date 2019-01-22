# -*- coding:utf-8 -*-
"""
@author Jacob
@time 2019/01/08
"""

import math
import numpy as np

"""
生成二维数据样本及可视化
plot 1-(0,0): circle_data
plot 2-(0,1): xor data
plot 3-(1,0): two-gauss data
plot 4-(1,1): spiral data
"""


# 随机洗牌，打乱顺序
# 实际之后没有使用
def shuffle(arr):
    np.random.shuffle(arr)
    return arr


class ScaleLinear(object):
    def __init__(self, domain, range_):
        """

        :param domain: x的序列
        :param range_: y的序列
        """
        self._domain = domain
        self._range = range_

    def __call__(self, value):
        """
        override __call__函数，实例被调用可直接计算线性插值
        :param value: 待估计的x（序列），估计方法为按照已知的_domain, _range进行线性插值
        :return: 返回插值
        """
        return np.interp(value, self._domain, self._range)


# ScaleLinear别名
linear = ScaleLinear


def dist(a, b):
    """
    求两点欧氏距离
    :param a: 待求距离的点坐标序列
    :param b: 待求距离的点的坐标序列
    :return: 点a和b的欧式距离
    """
    dx = a['x'] - b['x']
    dy = a['y'] - b['y']
    return math.sqrt(dx ** 2 + dy ** 2)


def random_normal(mean, variance):
    """
    生成给定均值和方差的正态分布随机数，不指定size。默认生成一个
    :param mean: 生成分布的均值
    :param variance: 生成分布的方差
    :return: 生成的对应分布随机数
    """
    return np.random.normal(mean, variance)


def random_uniform(a, b):
    """
    生成服从指定均匀分布的随机数
    :param a: 区间做左端点
    :param b: 区间右端点
    :return: 生成对应分布随机数
    """
    return np.random.uniform(a, b)


def classify_two_gauss_data(num_samples, noise):
    """
    plot 3的绘制方法：对称中心+给定方差的正态噪声
    :param num_samples: 样本个数，应当设为偶数
    :param noise: 噪声率
    :return: 生成的样本点dict{"x", "y", "label"}
    """
    points = []
    # 按照noise的尺度（在0~0.5范围内）线性生成扰动的方差（0.5~4）
    variance_scale = linear([0, .5], [0.5, 4])
    variance = variance_scale(noise)  # 如果noise=0.2,插值variancc=1.9

    def gen_gauss(cx, cy, label):
        """
        给定噪声方差，生成给定均值的正态随机数用于x和y坐标，不同的label对应不同的(x,y)中心（均值）点
        :param cx: X坐标均值
        :param cy: Y坐标均值
        :param label: 数据点的标签
        :return:
        """
        for _ in range(math.ceil(num_samples / 2)):
            x = random_normal(cx, variance**0.5)
            y = random_normal(cy, variance**0.5)
            points.append({'x': x, 'y': y, 'label': label})
    # label=1以（2,2）为中心
    # label=-1以（-2，-2）为中心
    gen_gauss(2, 2, 1)
    gen_gauss(-2, -2, -1)
    return points


def regress_plane(num_samples, noise):
    """
    实际这个函数没有使用
    :param num_samples: 样本量
    :param noise: 噪声率
    :return: 生成样本
    """
    radius = 6
    label_scale = linear([-10, 10], [-1, 1])

    def get_label(a, b):
        return label_scale(a + b)

    points = []

    for _ in range(num_samples):
        x = random_uniform(-radius, radius)
        y = random_uniform(-radius, radius)
        noise_x = random_uniform(-radius, radius) * noise
        noise_y = random_uniform(-radius, radius) * noise
        label = get_label(x + noise_x, y + noise_y)
        points.append({'x': x, 'y': y, 'label': label})

    return points


def regress_gaussian(num_samples, noise):
    """
    实际这个方法也没有使用
    :param num_samples:
    :param noise:
    :return:
    """
    points = []

    label_scale = linear([0, 2], [1, 0])

    gaussians = [
        [-4, 2.5, 1],
        [0, 2.5, -1],
        [4, 2.5, 1],
        [-4, -2.5, -1],
        [0, -2.5, 1],
        [4, -2.5, -1]
    ]

    def get_label(a, b):
        _label = 0
        for i in gaussians:
            new_label = i[2] * label_scale(dist({'x': a, 'y': b}, {'x': i[0], 'y': i[1]}))
            if abs(new_label) > abs(_label):
                _label = new_label
        return _label

    radius = 6

    for _ in range(num_samples):
        x = random_uniform(-radius, radius)
        y = random_uniform(-radius, radius)
        noise_x = random_uniform(-radius, radius) * noise
        noise_y = random_uniform(-radius, radius) * noise
        label = get_label(x + noise_x, y + noise_y)
        points.append({'x': x, 'y': y, 'label': label})

    return points


def classify_spiral_data(num_samples, noise):
    """
    plot 4的绘制方法：圆的极坐标+圆心角、半径增长+均匀分布噪声
    :param num_samples: 样本量
    :param noise: 噪声率
    :return: 生成样本
    """
    points = []
    n = num_samples / 2

    def gen_spiral(delta_t, label):
        for i in range(math.ceil(n)):
            r = i / n * 5  # 半径从0~5线性增长
            t = 1.75 * i / n * 2 * math.pi + delta_t  # 不包括偏移量的角度变化范围0~3.5π
            x = r * math.sin(t) + random_uniform(-1, 1) * noise  # 控制x和y的噪声小范围（窄带）波动，故不能使用正态噪声（无界）
            y = r * math.cos(t) + random_uniform(-1, 1) * noise
            points.append({'x': x, 'y': y, 'label': label})

    gen_spiral(0, 1)
    gen_spiral(math.pi, -1)

    return points


def classify_circle_data(num_samples, noise):
    """
    plot 1绘制方法: 圆和圆环+半径区分+均匀分布噪声
    :param num_samples: 样本量
    :param noise: 噪声率
    :return:
    """
    points = []
    radius = 5

    def get_circle_label(p, center):
        return 1 if dist(p, center) < radius * .5 else -1

    def gen_points(a, b):
        for _ in range(math.ceil(num_samples / 2)):
            # 给定半径的圆的内点
            r = random_uniform(a, b)
            angle = random_uniform(0, 2 * math.pi)
            x = r * math.sin(angle)
            y = r * math.cos(angle)
            noise_x = random_uniform(-radius, radius) * noise
            noise_y = random_uniform(-radius, radius) * noise
            label = get_circle_label({'x': x + noise_x, 'y': y + noise_y}, {'x': 0, 'y': 0})
            points.append({'x': x, 'y': y, 'label': label})

    # 在noise=0.2条件下，label为1的点半径不超过6，而label=-1的点半径不小于6，可以准确分离
    gen_points(0, radius * .5)
    gen_points(radius * .7, radius)

    return points


def classify_xor_data(num_samples, noise):
    """
    plot 2绘制方法：
    :param num_samples: 样本量
    :param noise: 噪声率
    :return: 生成样本点
    """
    def get_xor_label(p):
        return 1 if p['x'] * p['y'] >= 0 else -1

    points = []

    for _ in range(num_samples):
        # 将样本进行区块划分：
        # 左上角分块的右下顶点（-0.3,0.3）
        # 右上角分块的左下顶点（0.3,0.3）
        # 左下角分块的右上顶点（-0.3,-0.3）
        # 右下角分块的左上顶点（0.3，-0.3）
        # 引入噪声后会有干扰项
        padding = .3
        x = random_uniform(-5, 5)
        x += padding if x > 0 else -padding
        y = random_uniform(-5, 5)
        y += padding if y > 0 else -padding
        noise_x = random_uniform(-5, 5) * noise
        noise_y = random_uniform(-5, 5) * noise
        label = get_xor_label({'x': x + noise_x, 'y': y + noise_y})
        points.append({'x': x, 'y': y, 'label': label})

    return points


# playground拆包和重组接口
def get_samples(data_type, size, noise, clip=True):
    raw_data = data_type(size, noise)
    x1 = [r["x"] for r in raw_data]
    x2 = [r["y"] for r in raw_data]
    lab = [r["label"] for r in raw_data]
    data = np.array(list(zip(x1, x2)))
    if clip:
        label = np.clip(np.array(lab), 0.0, 1.0).reshape(-1, 1)
    else:
        label = np.array(lab).reshape(-1, 1)
    return data, label


if __name__ == '__main__':
    # from pprint import pprint
    #
    # foo = classify_xor_data(100, 1)
    # pprint(foo)

    import pandas as pd
    import matplotlib.pylab as plt
    import seaborn as sns


    def plot_test(plot_type, ratio, noise):
        s = int(1000 * ratio)
        pts = plot_type(s, noise)
        df = pd.DataFrame(pts)
        sns.scatterplot(x='x', y='y', hue='label', data=df)
        # plt.show()


    # plot_test(classify_circle_data, .8, 0.2)
    # plot_test(classify_xor_data, .8, 0.2)
    # plot_test(classify_two_gauss_data, .8, 0.2)
    # plot_test(classify_spiral_data, .8, 0.2)

    # 修改以使能绘制多图
    plt.figure()
    plt.subplot2grid((2, 2), (0, 0), colspan=1, rowspan=1)
    plot_test(classify_circle_data, .8, 0.2)
    plt.subplot2grid((2, 2), (0, 1), colspan=1, rowspan=1)
    plot_test(classify_xor_data, .8, 0.2)
    plt.subplot2grid((2, 2), (1, 0), colspan=1, rowspan=1)
    plot_test(classify_two_gauss_data, .8, 0.2)
    plt.subplot2grid((2, 2), (1, 1), colspan=1, rowspan=1)
    plot_test(classify_spiral_data, .8, 0.2)
    plt.show()
