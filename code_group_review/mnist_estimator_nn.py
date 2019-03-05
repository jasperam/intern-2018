# -*- coding: utf-8 -*-
# ! python3

"""
tensorflow.Estimator框架训练3层DNN网络用语MNIST分类
"""

import tensorflow as tf
import input_data
import numpy as np
import argparse
from scipy.ndimage import interpolation
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--pp_size", default=0, type=int,
                    help="preprocessing size to data augmentation")
parser.add_argument("--show", default=False, type=bool,
                    help="whether to show an example of random affine transformation")
parser.add_argument("--sigma", default=0.06, type=float,
                    help="affine transformation to control the gaussian noise in rotation and scaling")
parser.add_argument("--max_translation", default=3, type=int,
                    help="translation or shifting parameters in affine transformation matrix")


# 数据读取路径
TRAIN_DIR = "MNIST_data/"
BATCH_SIZE = 100
EPOCHS = 15
TRAIN_STEPS = 5e3
EVAL_STEPS = 500

# ------------------------------------------------------------
# read data and data preprocessing
# ------------------------------------------------------------
# 读取整个数据集，不设置validation set
mnist = input_data.read_data_sets(train_dir=TRAIN_DIR, validation_size=0)


def main(argv):
    args = parser.parse_args(argv[1:])

    def get_random_affine_matrix(sigma, max_translation):
        """
        Get a random affine transformation matrix
        Parameters
        ----------
        sigma: float
            Parametrizes the gaussian noise added to transformation matrix
        max_translation:
            Maximum translation allowed in the transformation matrix
        Returns
        -------
        matrix: ndarray
            3x3 matrix representing a random affine transformation
        """
        R = np.eye(2) + sigma * np.random.normal(0, 1, size=[2, 2])  # rotation and scale
        T = np.random.uniform(-max_translation, max_translation, size=2)  # translation
        return R, T

    def preprocessing(size, show):
        """
        数据预处理和数据增强
        :param size: 需要进行变换得到的新增数据个数；0则为不进行变换
        :param show: 是否展示仿射变换后的图像示例
        :return:
        """
        if size > 0:
            train_x = mnist.train.images
            train_y = mnist.train.labels
            new_x = []
            new_y = []
            # 除原本60000个训练样本外，另外通过仿射变换增强40000个样本；将训练集扩展到100,000
            print("Preprocessing: Affine transformation; total size {}".format(args.pp_size))
            for i in tqdm(np.random.permutation(len(train_x))[:40000]):
                img = np.reshape(train_x[i], (28, 28))
                mat, offset = get_random_affine_matrix(args.sigma, args.max_translation)
                trans = interpolation.affine_transform(img, matrix=mat, offset=offset)
                trans = np.reshape(trans, 784)
                new_x.append(trans)
                new_y.append(train_y[i])
                if show:
                    _, ax = plt.subplots(nrows=1, ncols=2, sharex="all", sharey="all")
                    ax = ax.flatten()
                    ax[0].set_xticks([])
                    ax[0].set_yticks([])
                    plt.tight_layout()
                    ax[0].imshow(img, cmap="Greys", interpolation="nearest")
                    ax[1].imshow(trans.reshape((28, 28)), cmap="Greys", interpolation="nearest")
                    plt.show()
                    plt.ion()
                    plt.pause(2)
                    plt.close()
                    show = False
            # 完成预处理后，再次同时打乱图像和标签的顺序
            train_x = np.concatenate([train_x, new_x])
            train_y = np.concatenate([train_y, new_y])
            train_x, train_y = shuffle(train_x, train_y, random_state=0)
            return (train_x, train_y), (mnist.test.images, mnist.test.labels)
        else:
            return (mnist.train.images, mnist.train.labels), \
               (mnist.test.images, mnist.test.labels)

    # 在不使用distortion的情况下
    (train_x, train_y), (test_x, test_y) = preprocessing(args.pp_size, args.show)

    # feature keys
    # 每个pixel作为一个feature，那么共有0~783个
    feature_keys = ["p" + str(i) for i in range(784)]

    # 转换train_x和test_x为dict
    train_x = dict(zip(feature_keys, np.transpose(train_x)))
    test_x = dict(zip(feature_keys, np.transpose(test_x)))

    # ----------------------------------------------------------
    # data input functions
    # ----------------------------------------------------------
    def train_input_fn(features, labels, batch_size):
        """
        input function for training
        :param features: x/data
        :param labels: y
        :param batch_size: number of samples in one mini-batch
        :return: Dataset
        """
        # convert to dataset with input as tuple
        labels = tf.cast(labels, dtype=tf.int32)
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
        # shuffle, buffer, repeat and batch slicing
        dataset = dataset.shuffle(1000, reshuffle_each_iteration=False).repeat().batch(batch_size)
        return dataset

    def eval_input_fn(features, labels, batch_size):
        """An input function for evaluation or prediction"""
        features = dict(features)
        if labels is None:
            # No labels, use only features.
            inputs = features
        else:
            labels = tf.cast(labels, dtype=tf.int32)
            inputs = (features, labels)

        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices(inputs)

        # Batch the examples
        assert batch_size is not None, "batch_size must not be None"
        dataset = dataset.shuffle(1000, reshuffle_each_iteration=False).repeat().batch(batch_size)

        # Return the dataset.
        return dataset

    # define feature columns
    feature_columns = [tf.feature_column.numeric_column(key=k) for k in feature_keys]

    # Build 2 hidden layer DNN with 300, 100 units
    classifier = tf.estimator.DNNClassifier(
        feature_columns=feature_columns,
        hidden_units=[300, 100],
        n_classes=10,
        activation_fn=tf.nn.relu,
        optimizer=lambda: tf.train.GradientDescentOptimizer(
            learning_rate=tf.train.piecewise_constant(
                x=tf.train.get_global_step(),
                boundaries=[2000, 5000, 8000],
                values=[1e-3, 5e-4, 2e-4, 1e-4]
            )
        ),
        config=tf.estimator.RunConfig(
            model_dir="./model_log",
            save_summary_steps=100,
            save_checkpoints_steps=500
        )
    )
    # # train the model
    # classifier.train(
    #     input_fn=lambda: train_input_fn(train_x,
    #                                     train_y,
    #                                     BATCH_SIZE),
    #     steps=TRAIN_STEPS
    # )
    # # eval the model
    # eval_result = classifier.evaluate(
    #     input_fn=lambda: eval_input_fn(test_x,
    #                                    test_y,
    #                                    BATCH_SIZE),
    #     steps=EVAL_STEPS
    # )
    # print("\nTest set accuracy: {accuracy:0.4f}\n".format(**eval_result))
    train_spec = tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(test_x,
                                                                test_y,
                                                                BATCH_SIZE),
                                        max_steps=TRAIN_STEPS*2)
    eval_spec = tf.estimator.EvalSpec(input_fn=lambda: eval_input_fn(test_x,
                                                             test_y,
                                                             BATCH_SIZE),
                                      start_delay_secs=600,
                                      throttle_secs=60)
    tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)

