
# coding: utf-8



import os
import tensorflow as tf
import sys
import urllib
from datetime import datetime

from urllib.request import urlretrieve

TRAIN_DIR = r"~\MNIST_data"

# -----------------------------------------------
# import dataset of Mnist: train, validation and test sets
# -----------------------------------------------
mnist_train = tfds.load(name="mnist", data_dir=TRAIN_DIR, split=tfds.Split.TRAIN, download=True)
mnist_test = tfds.load(name="mnist", data_dir=TRAIN_DIR, split=tfds.Split.TEST, download=True)

LOGDIR = '../tensorboard/cnn/v3.1/'
GITHUB_URL ='https://raw.githubusercontent.com/mamcgrath/TensorBoard-TF-Dev-Summit-Tutorial/master/'


### MNIST EMBEDDINGS ###
mnist = tf.contrib.learn.datasets.mnist.read_data_sets(train_dir=LOGDIR + 'data', one_hot=True)
### Get a sprite and labels file for the embedding projector ###
urlretrieve(GITHUB_URL + 'labels_1024.tsv', LOGDIR + 'labels_1024.tsv')
urlretrieve(GITHUB_URL + 'sprite_1024.png', LOGDIR + 'sprite_1024.png')


# + 卷积网络层的定义,tf.nn.conv2d是TensorFlow里面实现卷积的函数（读者也可以自己实现卷积操作）。其中：
#     + x是输入，即训练时输入的每一个batch。
#     + W是CNN中的卷积核，它要求是一个Tensor，具有[filter_height, filter_width, in_channels, out_channels]这样的shape，具体含义是[卷积核的高度，卷积核的宽度，图像通道数，卷积核个数]，要求类型与参数input相同。(图像通道即颜色的维度，卷积核个数代表卷积核有几层)
#     + strides：卷积时在图像每一维的步长，一个一维的向量，本例中每个步长为1. [1, 1, 1, 1]
#     + padding：string类型的量，只能是"SAME","VALID"其中之一，这个值决定了不同的卷积方式，其中SAME为零填充保持大小不变，VALID没有零填充
# 
# 
# + max pooling 层的定义,其中：
#     + x 是需要池化的输入，一般池化层接在卷积层后面，所以输入通常是feature map，依然是[batch, height, width, channels]这样的shape。
#     + ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
#     + strides：和卷积层的定义类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
#     + padding：和卷积层的定义类似，可以取'VALID' 或者'SAME'
#     + 返回一个Tensor，类型不变，shape是[batch, height, width, channels]的形式
#     
#     
# + padding的形式："VALID", "FULL", "SAME":
#     + VALID：比较容易理解，filter全部在image里面
#     <img src="image\VALID.png" width="30%" hegiht="30%" ></img>
#     + FULL(函数中没有)：足够的零填充，使得每个像素在每个方向上恰好被访问了k次（核的大小）
#     <img src="image\FULL.png" width="30%" hegiht="30%" ></img>
#     + SAME：满足$n_{out}=\lceil \frac {n_{in}} {s}\rceil$，即当步长s为1时，大小保持不变；各个方向补充0的规则为
#     <img src="image\SAME.png" width="30%" hegiht="30%" ></img>
#         - $pad_h = max[( n_{out} -1 ) × s_h + f_h - n_{in} ， 0]$
#         - $pad_{top} = \lfloor pad_h / 2 \rfloor$  # 注意此处向下取整
#         - $pad_{bottom} = pad_h - pad_{top}$
#         - $pad_w = max[( n_{out} -1 ) × s_w + f_w - n_{in} ， 0]$
#         - $pad_{left} = \lfloor pad_w / 2 \rfloor$  # 注意此处向下取整
#         - $pad_{right} = pad_w - pad_{left}$
#             
#             
# + [padding的图解](https://github.com/vdumoulin/conv_arithmetic)


# Add convolution layer
def conv_layer(input, size_in, size_out, name="conv"):
  with tf.name_scope(name): #设立名字域，以参数name命名
    w = tf.Variable(tf.truncated_normal([5, 5, size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding="SAME") #第一层卷积层，零填充使得卷积后大小不变(步长为1时)
    act = tf.nn.relu(conv + b) #第二层激活层，b用在cnv2d函数外面
    tf.summary.histogram("weights", w)#tensorboard中显示直方图
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return tf.nn.max_pool(act, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")#第三次池化层，最大池化


# Add fully connected layer 输出层不应该加激活函数
def fc_layer(input, size_in, size_out, name="fc"):
  with tf.name_scope(name):
    w = tf.Variable(tf.truncated_normal([size_in, size_out], stddev=0.1), name="W")
    b = tf.Variable(tf.constant(0.1, shape=[size_out]), name="B")
    act = tf.nn.relu(tf.matmul(input, w) + b)
    tf.summary.histogram("weights", w)
    tf.summary.histogram("biases", b)
    tf.summary.histogram("activations", act)
    return act


# In[5]:


def mnist_model(learning_rate, use_two_conv, use_two_fc, hparam):
  tf.reset_default_graph()
  sess = tf.Session()
  TIMESTAMP = ",{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now()) #增加时间路径，使得多次运行不重合

  # Setup placeholders, and reshape the data
  x = tf.placeholder(tf.float32, shape=[None, 784], name="x")
  x_image = tf.reshape(x, [-1, 28, 28, 1]) #-1表示使得总乘数不变的数
  tf.summary.image('input', x_image, 3)
  y = tf.placeholder(tf.float32, shape=[None, 10], name="labels")

  if use_two_conv:
    conv1 = conv_layer(x_image, 1, 32, "conv1") #输出为14*14，32个卷积核
    conv_out = conv_layer(conv1, 32, 64, "conv2") #输出为7*7，64个卷积核
  else:
    #conv1 = conv_layer(x_image, 1, 64, "conv") #输出为14*14，64个卷积核
    #conv_out = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME") #池化了两次？可去掉？
    #不可去掉，有必要，输出为7*7，64个卷积核
    
    #另一种做法：
    conv_out = conv_layer(x_image, 1, 16, "conv") #输出为14*14，16个卷积核，其中14*14*16=7*7*64
    
  flattened = tf.reshape(conv_out, [-1, 7 * 7 * 64]) #将数据由7*7*64变成1维


  if use_two_fc:
    fc1 = fc_layer(flattened, 7 * 7 * 64, 1024, "fc1") #将7*7*64维变成1024维
    embedding_input = fc1  #embedding输入,即最后一层的输入
    embedding_size = 1024  #embedding数据维度
    logits = fc_layer(fc1, 1024, 10, "fc2") #将1024维变成10维
  else:
    embedding_input = flattened
    embedding_size = 7*7*64
    logits = fc_layer(flattened, 7*7*64, 10, "fc") #将7*7*64维变成10维

  with tf.name_scope("xent"): #交叉熵
    xent = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=y), name="xent")
    tf.summary.scalar("xent", xent)

  with tf.name_scope("train"): #优化
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(xent)

  with tf.name_scope("accuracy"): #准确率
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar("accuracy", accuracy)

  summ = tf.summary.merge_all()


  #默认情况下，embedding projector 会用 PCA 主成分分析方法将高维数据投影到 3D 空间, 还有一种投影方法是 T-SNE。
  embedding = tf.Variable(tf.zeros([1024, embedding_size]), name="test_embedding")
  assignment = embedding.assign(embedding_input)
  saver = tf.train.Saver()

  sess.run(tf.global_variables_initializer())
  writer = tf.summary.FileWriter(LOGDIR + hparam + TIMESTAMP) #由summary,可导出tensorboard --logdir ../tensorboard/cnn/v3.1
  writer.add_graph(sess.graph)
  
  config = tf.contrib.tensorboard.plugins.projector.ProjectorConfig()
  embedding_config = config.embeddings.add()
  embedding_config.tensor_name = embedding.name
  embedding_config.sprite.image_path = '../sprite_1024.png'
  embedding_config.metadata_path = '../labels_1024.tsv'
  # Specify the width and height of a single thumbnail.
  embedding_config.sprite.single_image_dim.extend([28, 28])
  tf.contrib.tensorboard.plugins.projector.visualize_embeddings(writer, config)

  for i in range(2001):
    batch = mnist.train.next_batch(100) 
    if i % 5 == 0: #每5步记录一次tensorboard的数据summary
      [train_accuracy, s] = sess.run([accuracy, summ], feed_dict={x: batch[0], y: batch[1]}) #batch[0]为x图片数据，batch[1]为y标签
      writer.add_summary(s, i)
    if i % 500 == 0: #每500步保存一次模型
      sess.run(assignment, feed_dict={x: mnist.test.images[:1024], y: mnist.test.labels[:1024]}) #测试集的前1024个数据作为embedding的输入
      saver.save(sess, os.path.join(LOGDIR+hparam+TIMESTAMP, "model.ckpt"), i)
    sess.run(train_step, feed_dict={x: batch[0], y: batch[1]}) #用小批量训练


# In[6]:


def make_hparam_string(learning_rate, use_two_fc, use_two_conv):
    conv_param = "conv=2" if use_two_conv else "conv=1"
    fc_param = "fc=2" if use_two_fc else "fc=1"
    return "lr_%.0E,%s,%s" % (learning_rate, conv_param, fc_param)

def main():
  # You can try adding some more learning rates
  for learning_rate in [1E-3,1E-4]:

    # Include "False" as a value to try different model architectures
    for use_two_fc in [True,False]:
        for use_two_conv in [True,False]:
            # Construct a hyperparameter string for each one (example: "lr_1E-3,fc=2,conv=2)
            hparam = make_hparam_string(learning_rate, use_two_fc, use_two_conv)
            print('Starting run for %s' % hparam)

            # Actually run with the new settings
            mnist_model(learning_rate, use_two_fc, use_two_conv, hparam)


# In[7]:


if __name__ == '__main__':
  main()


# In[8]:


print(LOGDIR)


# In[9]:


mnist.test.labels[:1]


# In[ ]:




