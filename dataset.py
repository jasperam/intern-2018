"""
@author Jacob
@time 2019/01/08
"""

import math
import numpy as np


def shuffle(arr):
    np.random.shuffle(arr)
    return arr


class ScaleLinear(object):
    def __init__(self, domain, range_):
        self._domain = domain
        self._range = range_

    def __call__(self, value):
        return np.interp(value, self._domain, self._range)


linear = ScaleLinear


def dist(a, b):
    dx = a['x'] - b['x']
    dy = a['y'] - b['y']
    return math.sqrt(dx ** 2 + dy ** 2)


def random_normal(mean, variance):
    return np.random.normal(mean, variance)


def random_uniform(a, b):
    return np.random.uniform(a, b)


def classify_two_gauss_data(num_samples, noise):
    points = []
    variance_scale = linear([0, .5], [0.5, 4])
    variance = variance_scale(noise)
    v_sqrt = math.sqrt(variance)

    def gen_gauss(cx, cy, label):
        for _ in range(math.ceil(num_samples / 2)):
            x = random_normal(cx, v_sqrt)
            y = random_normal(cy, v_sqrt)
            points.append({'x': x, 'y': y, 'label': label})

    gen_gauss(2, 2, 1)
    gen_gauss(-2, -2, -1)
    return points


def regress_plane(num_samples, noise):
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
    points = []
    n = num_samples / 2

    def gen_spiral(delta_t, label):
        for i in range(math.ceil(n)):
            r = i / n * 5
            t = 1.75 * i / n * 2 * math.pi + delta_t
            x = r * math.sin(t) + random_uniform(-1, 1) * noise
            y = r * math.cos(t) + random_uniform(-1, 1) * noise
            points.append({'x': x, 'y': y, 'label': label})

    gen_spiral(0, 1)
    gen_spiral(math.pi, -1)

    return points


def classify_circle_data(num_samples, noise):
    points = []
    radius = 5

    def get_circle_label(p, center):
        return 1 if dist(p, center) < radius * .5 else -1

    def gen_points(a, b):
        for _ in range(math.ceil(num_samples / 2)):
            r = random_uniform(a, b)
            angle = random_uniform(0, 2 * math.pi)
            x = r * math.sin(angle)
            y = r * math.cos(angle)
            noise_x = random_uniform(-radius, radius) * noise
            noise_y = random_uniform(-radius, radius) * noise
            label = get_circle_label({'x': x + noise_x, 'y': y + noise_y}, {'x': 0, 'y': 0})
            points.append({'x': x, 'y': y, 'label': label})

    gen_points(0, radius * .5)
    gen_points(radius * .7, radius)

    return points


def classify_xor_data(num_samples, noise):
    def get_xor_label(p):
        return 1 if p['x'] * p['y'] >= 0 else -1

    points = []

    for _ in range(num_samples):
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
        plt.show()


    plot_test(classify_circle_data, .8, 0.2)
    plot_test(classify_xor_data, .8, 0.2)
    plot_test(classify_two_gauss_data, .8, 0.2)
    plot_test(classify_spiral_data, .8, 0.2)
