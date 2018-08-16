import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# MNIST数据集相关参数
INPUT_NODE = 784                # 输入层的节点数，等于图片的像素
OUTPUT_NODE = 10                # 输出层的节点数，等于类别的数目。0-9这10个数字，所以是10

# 配置神经网络的参数
LAYER1_NODE = 100               # 第一层隐藏层节点数，此隐藏层有100个节点
LAYER2_NODE = 100               # 第二层隐藏层节点数，此隐藏层有100个节点
BATCH_NODE = 200                # batch的大小
LEARNING_RATE_BASE = 0.8        # 基础的学习率
LEARNING_RATE_DECAY = 0.99      # 学习率的衰减率
REGULARIZATION_RATE = 0.0001    # 正则化项在损失函数中的系数
TRAINING_STEPS = 30000          # 训练轮数
MOVING_AVERAGE_DECAY = 0.99     # 滑动平均衰减率


# 在这里定义了一个使用ReLU激活函数的四层全连接网络
# 通过两层隐藏层实现了多层网络结构
# 通过ReLU激活函数实现了去线性化
# 不支持滑动平均模型
def inference_without_avg(input_tensor, reuse=False):
    with tf.variable_scope('layer1', reuse=reuse):
            weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.1))
            layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2', reuse=reuse):
        weights = tf.get_variable("weights", [LAYER1_NODE, LAYER2_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [LAYER2_NODE], initializer=tf.constant_initializer(0.1))
        layer2 = tf.nn.relu(tf.matmul(layer1, weights) + biases)

    with tf.variable_scope('layer3', reuse=reuse):
        weights = tf.get_variable("weights", [LAYER2_NODE, OUTPUT_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
        layer3 = tf.nn.relu(tf.matmul(layer2, weights) + biases)

    return layer3


# 在这里定义了一个使用ReLU激活函数的四层全连接网络
# 通过两层隐藏层实现了多层网络结构
# 通过ReLU激活函数实现了去线性化
# 支持滑动平均模型
def inference_with_avg(input_tensor, avg_class, reuse=False):
    with tf.variable_scope('layer1', reuse=reuse):
            weights = tf.get_variable("weights", [INPUT_NODE, LAYER1_NODE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.1))
            layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights)) + avg_class.average(biases))

    with tf.variable_scope('layer2', reuse=reuse):
        weights = tf.get_variable("weights", [LAYER1_NODE, LAYER2_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [LAYER2_NODE], initializer=tf.constant_initializer(0.1))
        layer2 = tf.nn.relu(tf.matmul(layer1, avg_class.average(weights)) + avg_class.average(biases))

    with tf.variable_scope('layer3', reuse=reuse):
        weights = tf.get_variable("weights", [LAYER2_NODE, OUTPUT_NODE],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.1))
        layer3 = tf.nn.relu(tf.matmul(layer2, avg_class.average(weights)) + avg_class.average(biases))

    return layer3


# 给定神经网络的输入，计算神经网络的前向传播结果
# 通过传入avg_class实现对滑动平均模型的支持
def inference(input_tensor, avg_class=None, reuse=False):
    if avg_class is None:
        return inference_without_avg(input_tensor, reuse=reuse)
    else:
        return inference_with_avg(input_tensor, avg_class, reuse=reuse)

# 训练模型的整个过程
def train(mnist):

    # 首先定义输入数据，使用了placeholder
    x = tf.placeholder(tf.float32, [None, INPUT_NODE], 'x-input')
    y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], 'y-input')

    # 调用inference函数计算前向传播的结果(不带滑动平均的)
    y = inference(x)

    with tf.variable_scope("", reuse=True):
        # 获得隐藏层1的权重
        weight1 = tf.get_variable("layer1/weights")
        # 获得隐藏层2的权重
        weight2 = tf.get_variable("layer2/weights")
        # 获得输出层的权重
        weight3 = tf.get_variable("layer3/weights")

    # 定义一个储存训练轮数的变量，用于滑动平均模型和指数衰减的学习率
    global_step = tf.Variable(0, trainable=False)

    # 给定滑动平均衰减率和训练轮数的变量，初始化滑动平均类
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)

    # 在所有代表神经网络参数的变量上使用滑动平均，即所有tf.trainable_variables集合中的变量，为这些变量都生成一个影子变量
    variable_averages_op = variable_averages.apply(tf.trainable_variables())

    # 计算使用了滑动平均之后的前向传播结果，可用于验证训练效果
    average_y = inference(x, variable_averages, reuse=True)

    # 计算交叉熵
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # 对整个batch的交叉熵求平均
    cross_entropy_mean = tf.reduce_mean(cross_entropy)

    # 生成一个L2正则化损失函数
    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    # 计算模型的正则化损失
    regularization = regularizer(weight1) + regularizer(weight2) + regularizer(weight3)

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

    # 因为使用了滑动平均模型，所以在训练神经网络时，每过一遍数据既需要通过反向传播来更新神经网络中的参数，又需要
    # 更新每个参数的滑动平均值。为了一次完成多个操作，可以使用tf.group()来将两个计算合并起来执行
    train_op = tf.group(train_step, variable_averages_op)

    # 检验对于batch中的每个数据，预测结果是否等于标记值
    correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
    # 计算模型在这一组数据中的正确率
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # 开启一个会话，开始计算过程
    with tf.Session() as sess:

        # 对所有变量进行初始化
        tf.global_variables_initializer().run()

        # 准备验证数据，一般可以在神经网络的训练过程中通过验证数据来大致判断停止的条件和评判训练的效果
        validate_feed = {
            x: mnist.validation.images,
            y_: mnist.validation.labels
        }

        # 准备测试数据，作为模型训练结束之后的最终评价标准
        test_feed = {
            x: mnist.test.images,
            y_: mnist.test.labels
        }

        # 迭代地训练神经网络
        for i in range(TRAINING_STEPS):

            # 每过1000轮使用验证数据评价模型
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                print("After %d training step(s), validation accuracy using average model is %g" % (i, validate_acc))

            # 每轮都提取一个batch的数据，训练神经网络
            xs, ys = mnist.train.next_batch(BATCH_NODE)
            sess.run(train_op, feed_dict={x: xs, y_: ys})

        # 在训练结束之后，在测试数据上检测神经网络模型的最终正确率
        test_acc = sess.run(accuracy, feed_dict=test_feed)
        print("After %d training step(s), test accuracy using average model is %g" % (i, test_acc))


# 主程序入口
def main(args=None):
    mnist = input_data.read_data_sets("/PycharmProjects/TFDemo/data/MNIST", one_hot=True)
    train(mnist)


# TensorFlow提供的一个主程序入口，tf.app.run()会调用上面定义的main函数
if __name__ == '__main__':
    tf.app.run()
