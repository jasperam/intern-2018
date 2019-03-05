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
TRAIN_STEP = 2000


# ------------------------------------------------------------
# read data and data preprocessing
# ------------------------------------------------------------
# 读取整个数据集，不设置validation set
mnist = input_data.read_data_sets(train_dir=TRAIN_DIR, validation_size=0)


def cnn_model_fn(features, labels, mode):
    # 根据论文，将原本28*28尺寸的图片周围填充0变为34*34的图片
    input_layer = tf.reshape(features["x"], [-1, 34, 34, 1])
    # layer 1: conv layer 1:
    # filter size: 7*7, 50 filters
    # stride = 1
    # padding=valid
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[7, 7],
        padding="valid",
        activation=tf.nn.relu
    )
    # layer 2: max pooling layer 1
    # filter size: 2*2
    # stride = 2
    pool1 = tf.layers.max_pooling2d(
        inputs=conv1,
        pool_size=[2, 2],
        strides=2
    )
    # just to simplify the net
    # layer 3: conv layer 2
    # filter size: 5*5, 128 filters
    # strides = 1
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=128,
        kernel_size=[5, 5],
        padding="valid",
        activation=tf.nn.relu)
    # layer 4: max pooling layer 2
    # filter size: 2*2
    # stride = 2
    pool2 = tf.layers.max_pooling2d(
        inputs=conv2,
        strides=2,
        pool_size=[2, 2]
    )
    # layer 5: dense layer
    # nerons: 200
    pool2_flat = tf.reshape(pool2, [-1, np.product(pool2.shape.as_list()[1:])])
    fc1 = tf.layers.dense(
        inputs=pool2_flat,
        units=200,
        activation=tf.nn.relu
    )
    # dropout
    # keep_rate: 0.8
    dropout1 = tf.layers.dropout(
        inputs=fc1, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
    # layer 6: dense layer
    # neurons: 200
    fc2 = tf.layers.dense(
        inputs=dropout1,
        units=200,
        activation=tf.nn.relu
    )
    # dropout
    # keep_rate: 0.8
    dropout2 = tf.layers.dropout(
        inputs=fc2, rate=0.2, training=mode == tf.estimator.ModeKeys.TRAIN)
    # layer 7: output layer
    logits = tf.layers.dense(
        inputs=dropout2,
        units=10
    )

    # prediction mode
    predictions = {
        "classes": tf.argmax(input=logits,
                             axis=1),
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    summary_hook = tf.train.SummarySaverHook(
        save_steps=100,
        output_dir=OUTPUT_DIR,
        summary_op=tf.summary.merge_all()
    )
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=
                                                      tf.train.exponential_decay(
                                                          learning_rate=0.01,
                                                          global_step=tf.train.get_global_step(),
                                                          decay_steps=500,
                                                          decay_rate=0.96
                                                      ))
        train_op = optimizer.minimize(loss=loss,
                                      global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          train_op=train_op,
                                          training_hooks=[summary_hook])
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
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops,
        evaluation_hooks=[eval_hook]
    )


def main(argv):

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

    (train_x, train_y), (test_x, test_y) = preprocessing(3)
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
        model_fn=cnn_model_fn,
        model_dir=MODEL_DIR,
        config=tf.estimator.RunConfig(
            save_summary_steps=100,
            save_checkpoints_steps=500)
    )
    # classifier.train(
    #     input_fn=train_input_fn,
    #     # steps=TRAIN_STEP,
    #     max_steps=TRAIN_STEP
    # )
    # eval_results = classifier.evaluate(input_fn=eval_input_fn)
    # print(eval_results)
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
