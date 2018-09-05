import os
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

INPUT_NODE = 784                # 输入节点数
OUTPUT_NODE = 10                # 输出节点数

IMAGE_SIZE = 28                 # 图片的长和宽
NUM_CHANNELS = 1                # 图片的频道数，单色频道为1，多色RGB频道为3
NUM_LABELS = 10                 # 标记的数目

# 第一层卷积层的尺寸和深度
CONV1_DEEP = 32                 # 第一层卷积层的深度
CONV1_SIZE = 5                  # 第一层卷积层的尺寸

# 第二层卷积层的尺寸和深度
CONV2_DEEP = 64                 # 第二层卷积层的深度
CONV2_SIZE = 5                  # 第二层卷积层的尺寸

# 全连接层的节点数
FC_SIZE = 512

BATCH_NODE = 200                # batch的大小
LEARNING_RATE_BASE = 0.1        # 基础的学习率
LEARNING_RATE_DECAY = 0.99      # 学习率的衰减率
REGULARIZATION_RATE = 0.0001    # 正则化项在损失函数中的系数
TRAINING_STEPS = 30000          # 训练轮数

MODEL_SAVE_PATH = "/PycharmProjects/TFDemo/data/model/531/"
MODEL_NAME = "model.ckpt"


# 定义卷积神经网络的前向传播过程
# 这里添加了一个参数regularizer，用来为全连接层添加正则化损失
def inference(input_tensor, regularizer):

    # 声明第一层卷积层并实现前向传播过程
    # 此层的输入为28*28*1的原始MNIST图片像素
    # 因为卷积层使用了全0填充，所以输出为28*28*32的矩阵
    with tf.variable_scope('layer1-conv1'):

        # 定义和初始化第一层的权重和偏置
        conv1_weights = tf.get_variable("weight", [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable("bias", [CONV1_DEEP], initializer=tf.constant_initializer(0.0))

        # 使用边长为5，深度为32的过滤器，过滤器移动的步长为1，且使用全0填充
        conv1 = tf.nn.conv2d(input_tensor, filter=conv1_weights, strides=[1, 1, 1, 1], padding="SAME")

        # 使用ReLu作为激活函数实现去线性化
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))

    # 实现第二层池化层的前向传播过程
    with tf.name_scope('layer2-pool1'):

        # 这里使用的是最大池化层
        # 池化层过滤器的尺寸为2，使用全0填充并且步长为2
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 声明第三层卷积层的变量并实现前向传播过程，与第一层相似
    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable("weight", [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable("bias", [CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))

    # 实现第四层池化层的前向传播过程，与第二层相似
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

    # 将第四层池化层的输出转化为第五层全连接层的输入格式
    pool_shape = pool2.get_shape().as_list()
    # 计算出向量的长度
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    # -1代表的是此位置的参数根据实际情况给出，并不直接提供定值
    reshaped = tf.reshape(pool2, [-1, nodes])

    # 声明第五层全连接层的变量并实现前向传播过程
    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable("weight", [nodes, FC_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection("losses", regularizer(fc1_weights))
        fc1_biases = tf.get_variable("bias", [FC_SIZE], initializer=tf.constant_initializer(0.1))
        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)

    # 声明第六层全连接层的变量并实现前向传播过程
    with tf.variable_scope('layer6-fc2'):
        fc2_weights = tf.get_variable("weight", [FC_SIZE, NUM_LABELS], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer is not None:
            tf.add_to_collection('losses', regularizer(fc2_weights))
        fc2_biases = tf.get_variable("bias", [NUM_LABELS], initializer=tf.constant_initializer(0.1))
        logit = tf.matmul(fc1, fc2_weights) + fc2_biases

    return logit


# 训练过程
def train(mnist):

    # 首先定义输入数据，使用了placeholder
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS], 'x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], 'y-input')

    # 定义一个储存训练轮数的变量，用于滑动平均模型和指数衰减的学习率
    global_step = tf.Variable(0, trainable=False)

    # 生成一个L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)

    # 调用inference函数计算前向传播的结果
    y = inference(x, regularizer)

    # 计算交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 对整个batch的交叉熵求平均
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 将loss集合中所有元素相加，得到所有的正则化损失
    regularization = tf.add_n(tf.get_collection('losses'))

    # 将交叉熵和正则化损失求和，得到模型的总损失
    loss = cross_entropy_mean + regularization

    # 设置指数衰减的学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,                     # 基础的学习率，随着迭代的进行，在这个学习率的基础上递减
        global_step,                            # 之前定义的用于储存训练轮数的变量
        mnist.train.num_examples/BATCH_NODE,    # 衰减速度
        LEARNING_RATE_DECAY)                    # 衰减系数，若衰减速度为100，衰减系数为.96，则说明每过100轮学习率变成之前的.96

    # 使用梯度下降算法定义优化过程
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

    # 检验对于batch中的每个数据，预测结果是否等于标记值
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # 计算模型在这一组数据中的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 声明saver用于持久化训练模型
    saver = tf.train.Saver()

    # 开启一个会话，开始计算过程
    def calculate():
        with tf.Session() as sess:

            # 对所有变量进行初始化
            tf.global_variables_initializer().run()

            # 准备验证数据，一般可以在神经网络的训练过程中通过验证数据来大致判断停止的条件和评判训练的效果
            validate_xs, validate_ys = mnist.validation.next_batch(BATCH_NODE)
            reshaped_validate_xs = np.reshape(validate_xs, [BATCH_NODE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
            validate_feed = {
                x: reshaped_validate_xs,
                y_: validate_ys
            }

            # 准备测试数据，作为模型训练结束之后的最终评价标准
            test_xs = mnist.test.images
            reshaped_test_xs = np.reshape(test_xs, [mnist.test.num_examples, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])
            test_feed = {
                x: reshaped_test_xs,
                y_: mnist.test.labels
            }

            # 迭代地训练神经网络
            for i in range(TRAINING_STEPS):

                # 每过1000轮使用验证数据评价模型并对模型进行持久化
                if i % 1000 == 0:
                    validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                    # 使用saver对模型持久化
                    saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
                    print("After %d training step(s), validation accuracy using average model is %g" % (i, validate_acc))

                # 如果3000轮正确率还没到90%，就重新生成随机数，重新开始
                if i == 3000 and float(validate_acc) <= 0.9:
                    return False

                # 每轮都提取一个batch的数据，训练神经网络
                xs, ys = mnist.train.next_batch(BATCH_NODE)

                # 将输入的xs调整为四维矩阵，以训练卷积神经网络
                reshaped_xs = np.reshape(xs, [BATCH_NODE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS])

                sess.run(train_step, feed_dict={x: reshaped_xs, y_: ys})

            # 在训练结束之后，在测试数据上检测神经网络模型的最终正确率
            test_acc = sess.run(accuracy, feed_dict=test_feed)
            print("After %d training step(s), test accuracy using average model is %g" % (i, test_acc))
            return True

    flag = True
    while flag:
        flag = not calculate()


# 主程序入口
def main(args=None):
    mnist = input_data.read_data_sets("/PycharmProjects/TFDemo/data/MNIST", one_hot=True)
    train(mnist)


# TensorFlow提供的一个主程序入口，tf.app.run()会调用上面定义的main函数
if __name__ == '__main__':
    tf.app.run()


