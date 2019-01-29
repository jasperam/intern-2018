#!/usr/bin/python
# coding:utf-8


import tensorflow as tf
# Axes3D as base_axes object
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


class RegWithVisual:
    def __init__(self, x0, y0, reg_type="L1", reg_rate=0.03, step=40):
        self.graph = tf.Graph()
        self.x0 = x0
        self.y0 = y0
        self.reg_type = reg_type
        self.reg_rate = reg_rate
        self.plot_type = 'surface'
        self.step = step

    def set_x_y(self, init_x, init_y):
        with self.graph.as_default():
            x = tf.Variable(init_x, [1], dtype=tf.float32)
            y = tf.Variable(init_y, [1], dtype=tf.float32)
            return x, y

    def set_x0(self, x0):
        self.x0 = x0

    def set_y0(self, y0):
        self.y0 = y0

    def set_reg_type(self, reg_type):
        self.reg_type = reg_type

    def set_reg_rate(self, reg_rate):
        self.reg_rate = reg_rate

    def set_plot_type(self, plot_type):
        self.plot_type = plot_type

    def cost_func(self, x=None, y=None, cost_type=0):
        with self.graph.as_default():
            if not x:
                x = tf.placeholder(dtype=tf.float32, shape=[None, 1])
            if not y:
                y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
            # 抛物线函数
            cost = tf.add(tf.square(x - self.x0), tf.square(y - self.y0))
            # 正则化函数
            if cost_type == 1:
                cost = tf.add(tf.abs(x), tf.abs(y)) * self.reg_rate + cost
            elif cost_type == 2:
                cost = tf.add(tf.square(x), tf.square(y)) * self.reg_rate + cost
            return x, y, cost

    def set_plot(self, plot_one=True):
        plt.ion()
        fig = plt.figure(figsize=(3, 2), dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
        params = {'legend.fontsize': 3,
                  'legend.handlelength': 3}
        plt.rcParams.update(params)
        plt.axis('off')
        x_val = y_val = np.arange(-1.5, 1.5, 0.005, dtype=np.float32)
        x_val_mesh, y_val_mesh = np.meshgrid(x_val, y_val)
        x_val_mesh_flat = x_val_mesh.reshape([-1, 1])
        y_val_mesh_flat = y_val_mesh.reshape([-1, 1])
        cost_type = [1, 2][self.reg_type == "L2"] if plot_one else 0
        x, y, cost = self.cost_func(cost_type=cost_type)
        sess = tf.InteractiveSession(graph=self.graph)
        cost_val_mesh_flat = sess.run(cost, feed_dict={x: x_val_mesh_flat,
                                  y: y_val_mesh_flat})
        sess.close()
        cost_val_mesh = cost_val_mesh_flat.reshape(x_val_mesh.shape)
        if self.plot_type == "surface":
            ax.plot_surface(x_val_mesh,
                            y_val_mesh,
                            cost_val_mesh,
                            alpha=.4,
                            cmap=plt.cm.coolwarm)
            ax.view_init(elev=2, azim=-118)
        elif self.plot_type == "contour":
            levels = np.arange(-1, 10, 0.2)
            ax.contour(x_val_mesh,
                            y_val_mesh,
                            cost_val_mesh,
                            levels,
                            alpha=.4,
                            linewidths=0.4)
            azm = ax.azim
            ele = ax.elev + 40
            ax.view_init(elev=ele, azim=azm)
        plt.draw()
        plt.pause(0.001)
        xlm = ax.get_xlim3d()
        ylm = ax.get_ylim3d()
        zlm = ax.get_zlim3d()
        ax.set_xlim3d(xlm[0] * 0.5, xlm[1] * 0.5)
        ax.set_ylim3d(ylm[0] * 0.5, ylm[1] * 0.5)
        ax.set_zlim3d(zlm[0] * 0.5, zlm[1] * 0.5)
        return ax

    def plot_one(self, x, y, cost, optimizer, label, ax):
        with self.graph.as_default():
            training_op = optimizer.minimize(cost)
            init = tf.global_variables_initializer()
            sess = tf.InteractiveSession(graph=self.graph)
            sess.run(init)
            steps = self.step
            plot_cache = None
            last_z = last_x = last_y = None

            for itr in range(steps):
                _, x_val, y_val, cost_val = sess.run([training_op, x, y, cost])
                plot_cache = \
                    ax.scatter(x_val, y_val, cost_val,
                               s=3, depthshade=True, label=label,
                               color=['k', 'r'][self.reg_type == "L2"])
                if not itr:
                    last_z = cost_val
                    last_x = x_val
                    last_y = y_val
                    plt.legend([plot_cache], [label])
                ax.plot([last_x, x_val],
                        [last_y, y_val],
                        [last_z, cost_val],
                        linewidth=0.5, color=['k', 'r'][self.reg_type == "L2"])
                last_z = cost_val
                last_x = x_val
                last_y = y_val
                if not Path("figures").is_dir():
                    Path("figures").mkdir(parents=True)
                plt.savefig('figures/{}{:0>3d}_{}.png'.format(self.plot_type, itr, self.reg_type))
                print('iteration: {}'.format(iter))
                plt.pause(0.0001)
            sess.close()

    def plot_two(self, x_i, y_i, cost1, cost2, cost1_0, cost2_0, ax):
        with self.graph.as_default():
            ops = []
            ops.append(tf.train.GradientDescentOptimizer(0.05).minimize(cost1))
            ops.append(tf.train.GradientDescentOptimizer(0.05).minimize(cost2))
            init = tf.global_variables_initializer()
            sess = tf.InteractiveSession(graph=self.graph)
            sess.run(init)
            steps = self.step
            plot_cache = [None for _ in range(len(ops))]
            last_z, last_x, last_y = [], [], []

            for itr in range(steps):
                for i, op in enumerate(ops):
                    _, x_val, y_val, cost_val = \
                        sess.run([op, x_i[i], y_i[i], [cost1_0, cost2_0][i]])
                    if plot_cache[i]:
                        plot_cache[i].remove()
                    plot_cache[i] = \
                        ax.scatter(x_val, y_val, cost_val,
                                   s=3, depthshade=True,
                                   color=['k', 'r'][i])
                    if not itr:
                        last_z.append(cost_val)
                        last_x.append(x_val)
                        last_y.append(y_val)
                        plt.legend(plot_cache, ["GD_L1", "GD_L2"])
                    ax.plot([last_x[i], x_val],
                            [last_y[i], y_val],
                            [last_z[i], cost_val],
                            linewidth=0.5, color=['k', 'r'][i])
                    last_z[i] = cost_val
                    last_x[i] = x_val
                    last_y[i] = y_val
                if not Path("figures").is_dir():
                    Path("figures").mkdir(parents=True)
                plt.savefig('figures/{}{:0>3d}_L1L2.png'.format(self.plot_type, itr))
                print('iteration: {}'.format(iter))
                plt.pause(0.0001)
            sess.close()

    def plot_gif(self):
        import imageio
        from glob import glob
        image_files = {}
        contourl1 = glob("figures\contour[0-9]*_L1.png")
        contourl2 = glob("figures\contour[0-9]*_L2.png")
        contourl1l2 = glob("figures\contour[0-9]*_L1L2.png")
        surfacel1 = glob("figures\surface[0-9]*_L1.png")
        surfacel2 = glob("figures\surface[0-9]*_L2.png")
        surfacel1l2 = glob("figures\surface[0-9]*_L1L2.png")
        if contourl1:
            contourl1 = ["figures\contour{:0>3d}_L1.png".format(itr) for itr in range(self.step)]
        if contourl2:
            contourl2 = ["figures\contour{:0>3d}_L2.png".format(itr) for itr in range(self.step)]
        if contourl1l2:
            contourl1l2 = ["figures\contour{:0>3d}_L1L2.png".format(itr) for itr in range(self.step)]
        if surfacel1:
            surfacel1 = ["figures\surface{:0>3d}_L1.png".format(itr) for itr in range(self.step)]
        if surfacel2:
            surfacel2 = ["figures\surface{:0>3d}_L2.png".format(itr) for itr in range(self.step)]
        if surfacel1l2:
            surfacel1l2 = ["figures\surface{:0>3d}_L1L2.png".format(itr) for itr in range(self.step)]

        image_files.update({
            "contourl1": contourl1,
            "contourl2": contourl2,
            "contourl1l2": contourl1l2,
            "surfacel1": surfacel1,
            "surfacel2": surfacel2,
            "surfacel1l2": surfacel1l2
        })
        imageio.plugins.freeimage.download()
        for label, files in image_files.items():
            images = []
            for ff in files:
                images.append(imageio.imread(ff))
            if images:
                imageio.mimsave('./figures/{}.gif'.format(label),
                                images,
                                format="GIF",
                                duration=0.5)

    def reset_graph(self):
        self.graph = tf.Graph()


if __name__ == "__main__":
    obj = RegWithVisual(x0=-0.5, y0=-0.8)
    obj.set_reg_rate(0.2)
    x_1, y_1 = obj.set_x_y(0.75, 1.0)
    x_2, y_2 = obj.set_x_y(0.75, 1.0)
    obj.set_plot_type("contour")
    _, _, cost1 = obj.cost_func(x_1, y_1, 1)
    _, _, cost2 = obj.cost_func(x_2, y_2, 2)
    _, _, cost1_0 = obj.cost_func(x_1, y_1, 0)
    _, _, cost2_0 = obj.cost_func(x_2, y_2, 0)
    # ax = obj.set_plot(True)
    # obj.plot_one(x_1, y_1, cost1, tf.train.GradientDescentOptimizer(0.05), "GD_L1", ax)

    ax = obj.set_plot(False)
    obj.plot_two([x_1, x_2], [y_1, y_2], cost1, cost2, cost1_0, cost2_0, ax)
    obj.plot_gif()



