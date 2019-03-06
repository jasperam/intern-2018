# -*- coding: utf-8 -*-
# ! python3


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import input_data
import numpy as np


TRAIN_DIR = "MNIST_data/"
MODEL_DIR = "model_cnn_log"
OUTPUT_DIR = "model_cnn_log/train"
OUTPUT_EVAL_DIR = "model_cnn_log/eval"
BATCH_SIZE = 100
TRAIN_STEP = 6000


# ------------------------------------------------------------
# read data and data preprocessing
# ------------------------------------------------------------
# 读取整个数据集，不设置validation set
mnist = input_data.read_data_sets(train_dir=TRAIN_DIR, validation_size=0)


def simard_model_fn(features, labels, mode):
    """Model function of simple cnn from Simard(2003)"""
    # Input Layer, reshape the images to 29*29*1 according to the paper
    input_layer = tf.reshape(features["x"], [-1, 29, 29, 1])
    # For black-and-white photo with 29 * 29 pixels --mnist

    # Convolutional Layer #1
    # Filter size: 5*5, 5 filters
    # Stride = 2
    # Padding = valid
    # Convert 29*29*1 to 13*13*5
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        strides=2,
        filters=5,
        kernel_size=[5, 5],
        padding="valid",
        # kernel_initializer=tf.truncated_normal_initializer(stddev=0.005),
        activation=tf.nn.relu)

    # Convolutional Layers #2
    # Filter size: 5*5, 50 filters
    # Stride = 2
    # Padding = valid
    # Convert 13*13*5 to 5*5*50
    conv2 = tf.layers.conv2d(
        inputs=conv1,
        strides=2,
        filters=50,
        kernel_size=[5, 5],
        padding="valid",
        # kernel_initializer=tf.truncated_normal_initializer(stddev=0.005),
        activation=tf.nn.relu)

    # Dense Layer #3
    # Hidden units: 100
    conv2_flat = tf.reshape(conv2, [-1, 5 * 5 * 50])
    dense = tf.layers.dense(inputs=conv2_flat,
                            # kernel_initializer=tf.truncated_normal_initializer(stddev=0.005),
                            units=100, activation=tf.nn.relu)

    # Logits Layer #4
    # Output units: 10
    logits = tf.layers.dense(inputs=dense,
                             # kernel_initializer=tf.truncated_normal_initializer(stddev=0.005),
                             units=10)

    # Generate predictions (for PREDICT and EVAL mode)
    predictions = {
        "classes": tf.argmax(input=logits, axis=1),
        # Add 'softmax_tensor' to the graph. It is used for PREDICT and by the
        # 'logging_hook'
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # Prediction mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss using Cross-Entropy(for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    summary_hook = tf.train.SummarySaverHook(
        save_steps=100,
        output_dir=OUTPUT_DIR,
        summary_op=tf.summary.merge_all()
    )

    # Configure the Training Op (for TRAIN mode)
    lr = 0.05
    if mode == tf.estimator.ModeKeys.TRAIN:
        step = tf.train.get_or_create_global_step()
        # Learning rate is multiplied by 0.3 after every 1000 step
        boundaries = []
        values = [lr]
        for i in range(6):
            boundaries.append(i*1000)
            values.append(lr*(0.3**(i+1)))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=
                                                      tf.train.piecewise_constant(step, boundaries, values))
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, training_hooks=summary_hook)

    # Add evaluation metrics (for EVAL mode)
    acc_op = tf.metrics.accuracy(
        labels=labels, predictions=predictions["classes"],
        name="accuracy"
    )
    eval_metric_ops = {
        "accuracy": acc_op
    }
    acc_summary = tf.summary.scalar("accuracy", acc_op[0])
    eval_hook = tf.train.SummarySaverHook(
        save_steps=500,
        output_dir=OUTPUT_EVAL_DIR
    )

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, evaluation_hooks=[eval_hook])


def main(argv, prep = None):

    # Preprocessing 1 : add zeros to reshape the images to 29*29*1 according to the paper
    def prep1(size=28):
        train_x = mnist.train.images
        test_x = mnist.test.images
        image = []
        for mat in [train_x, test_x]:
            ret = np.zeros((mat.shape[0], mat.shape[1] + 1, mat.shape[2] + 1))
            for i in range(mat.shape[0]):
                temp = np.vstack((np.zeros((1, size)), mat[i]))
                image.append(np.hstack((temp, np.zeros((size + 1, 1)))))
        return (image[0], mnist.train.labels), (image[1], mnist.test.labels)

    # Preprocessing 2 : affine distortion
    def prep2():
        return 0

    # Preprocessing 3 : elastic deformation
    def prep3():
        return 0

    def preprocessing(size=3):
        train_x = mnist.train.images
        test_x = mnist.test.images
        if size:
            train_x = train_x.reshape([-1, 28, 28])
            test_x = test_x.reshape([-1, 28, 28])
            pad11 = np.zeros([train_x.shape[0], 28, size])
            pad12 = np.zeros([test_x.shape[0], 28, size])
            pad21 = np.zeros([train_x.shape[0], size, 34])
            pad22 = np.zeros([test_x.shape[0], size, 34])
            train_x = np.concatenate(
                [pad21, np.concatenate([pad11, train_x, pad11], axis=2), pad21], axis=1)
            test_x = np.concatenate(
                [pad22, np.concatenate([pad12, test_x, pad12], axis=2), pad22], axis=1)
            train_x = train_x.reshape((-1, 1156))
            test_x = test_x.reshape((-1, 1156))
            return (train_x, mnist.train.labels), (test_x, mnist.test.labels)

        return (mnist.train.images, mnist.train.labels), \
               (mnist.test.images, mnist.test.labels)

    (train_x, train_y), (test_x, test_y) = prep1(28)
    if prep == 'affine':
        (train_x, train_y), (test_x, test_y) = prep2()
    if prep == 'elastic':
        (train_x, train_y), (test_x, test_y) = prep3()
    train_y = np.asarray(train_y, dtype=np.int32)
    test_y = np.asarray(test_y, dtype=np.int32)
    # ----------------------------------------------------------
    # data input functions
    # ----------------------------------------------------------
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_x},
        y=train_y,
        batch_size=BATCH_SIZE,
        num_epochs=None,
        shuffle=True)
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": test_x},
        y=test_y,
        num_epochs=1,
        shuffle=False)
    # ----------------------------------------------------------
    # train and evaluate model
    # ----------------------------------------------------------
    classifier = tf.estimator.Estimator(
        model_fn=simard_model_fn,
        model_dir=MODEL_DIR,
        config=tf.estimator.RunConfig(
            save_summary_steps=100,
            save_checkpoints_steps=500)
    )

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn,
                                        max_steps=TRAIN_STEP)
    eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn,
                                      start_delay_secs=600,
                                      throttle_secs=60)
    eval_results = tf.estimator.train_and_evaluate(classifier, train_spec, eval_spec)
    print(eval_results)


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
