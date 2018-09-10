import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


# 声明常量
HIDDEN_SIZE = 30                    # LSTM中隐藏节点的个数
NUM_LAYERS = 2                      # LSTM的层数

TIMESTEPS = 10                      # 循环神经网络的训练序列长度
TRAINING_STEPS = 6000               # 训练轮数
BATCH_SIZE = 32                     # batch的大小

TRAINING_EXAMPLES = 10000           # 训练数据个数
TESTING_EXAMPLES = 1000             # 测试数据个数
SAMPLE_GAP = 0.1                    # 采样间隔


# 产生训练所用数据集X和标记Y
# 输入为一个数组,数组中的每个元素是sin函数的x值和对应的y值
# 输出为生成的数据集x和标记y
def generate_data(seq):
    x, y = [], []

    # 将序列的第i项到第i+TIMESTEPS-1项，共TIMESTEPS项作为输入
    # 将序列的第i+TIMESTEPS项作为输出
    # 即，使用sin函数前TIMESTEPS个节点的信息，来预测第i + TIMESTEPS个节点的函数值
    for i in range(len(seq) - TIMESTEPS):
        x.append([seq[i:i+TIMESTEPS]])
        y.append([seq[i + TIMESTEPS]])

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)


# 定义模型、前向传播过程、损失函数和优化过程
# 输入为数据x、标记y和是否是训练过程is_training
# 返回预测结果，损失函数，优化过程
def lstm_model(x, y, is_training):

    # 使用多层的LSTM结构
    cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(HIDDEN_SIZE) for _ in range(NUM_LAYERS)])

    # 将多层的LSTM结构连接成RNN网络并计算其前向传播的结果
    outputs, _ = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    # outputs是顶层LSTM在每一步的输出结果
    # 它的维度为[batch_size, time, HIDDEN_SIZE]
    # 本问题中只关注最后一个时刻的输出结果
    output = outputs[:, -1, :]

    # 对LSTM网络的输出再加一层全连接层并计算损失
    predictions = tf.contrib.layers.fully_connected(output, 1, activation_fn=None)

    # 只在训练时计算损失函数和优化步骤，测试时直接返回预测结果
    if not is_training:
        return predictions, None, None

    # 计算损失函数
    loss = tf.losses.mean_squared_error(y, predictions)

    # 创建模型优化器并得到优化步骤
    train_op = tf.contrib.layers.optimize_loss(loss, tf.train.get_global_step(), optimizer="Adagrad", learning_rate=0.1)

    # 返回预测结果，损失函数，优化过程
    return predictions, loss, train_op


# 训练模型
def train(sess, train_x, train_y):

    # 对数据进行处理，将训练数据以数据集的方式提供给计算图

    # 切分传入Tensor的第一个维度，生成相应的dataSet
    ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))

    # ds.repeat : Repeats this dataset `count` times.
    # ds.shuffle : 随机打乱这个数据集的元素
    # ds.batch : 将此数据集的连续元素组合成batch
    ds = ds.repeat().shuffle(1000).batch(BATCH_SIZE)

    # make_one_shot_iterator()用来生成一个迭代器来读取数据
    # one_shot迭代器人如其名，意思就是数据输出一次后就丢弃了
    # 之后每次x, y被会话调用迭代器就会将指针指向下一个元素
    x, y = ds.make_one_shot_iterator().get_next()

    # 调用模型传入数据并得到预测结果、损失函数和优化步骤
    with tf.variable_scope("model"):
        predictions, loss, train_op = lstm_model(x, y, True)

    # 初始化变量
    sess.run(tf.global_variables_initializer())

    # 开启训练
    for i in range(TRAINING_STEPS):
        _, loss_result = sess.run([train_op, loss])
        if i % 100 == 0:
            print("After %d training step(s), the loss result is %f" % (i, loss_result))


# 测试模型
def run_eval(sess, test_x, test_y):

    # 将测试数据以数据集的方式提供给计算图

    # 切分传入Tensor的第一个维度，生成相应的dataSet
    data_set = tf.data.Dataset.from_tensor_slices((test_x, test_y))

    # 类型转换，将Dataset转换为BatchDataset
    data_set = data_set.batch(1)

    # make_one_shot_iterator()用来生成一个迭代器来读取数据
    # one_shot迭代器人如其名，意思就是数据输出一次后就丢弃了
    x, y = data_set.make_one_shot_iterator().get_next()

    with tf.variable_scope("model", reuse=True):

        # 调用模型得到预测结果
        prediction, _, _ = lstm_model(x, [0.0], False)

        # 将预测结果存入数组中
        predictions, labels = [], []
        for i in range(TESTING_EXAMPLES):
            prediction_result, labels_result = sess.run([prediction, y])
            predictions.append(prediction_result)
            labels.append(labels_result)

        # 计算rmse作为评价标准
        predictions = np.array(predictions).squeeze()
        labels = np.array(labels).squeeze()
        rmse = np.sqrt(((predictions - labels) ** 2).mean(axis=0))
        print("Mean Square Error is: %f" % rmse)

        # 对预测的sin函数曲线进行绘图
        plt.figure()
        plt.plot(labels, label="real_sin")
        plt.plot(predictions, label='predictions')
        plt.legend()
        plt.show()


# 主程序入口
def main(args=None):

    # 产生训练集数据的起始值
    train_start = 0
    # 产生训练集数据的终止值，也就是测试集数据的起始值
    train_end = test_start = (TRAINING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP
    # 产生测试集数据的终止值
    test_end = test_start + (TESTING_EXAMPLES + TIMESTEPS) * SAMPLE_GAP

    # 产生训练集数据和测试集数据
    train_x, train_y = generate_data(np.sin(np.linspace(train_start, train_end, TRAINING_EXAMPLES + TIMESTEPS, dtype=np.float32)))
    test_x, test_y = generate_data(np.sin(np.linspace(test_start, test_end, TESTING_EXAMPLES + TIMESTEPS, dtype=np.float32)))

    # 开启会话开始计算
    with tf.Session() as sess:

        # 训练模型
        train(sess, train_x, train_y)

        # 使用训练好的模型对测试数据进行预测
        run_eval(sess, test_x, test_y)


# TensorFlow提供的一个主程序入口，tf.app.run()会调用上面定义的main函数
if __name__ == '__main__':
    tf.app.run()

# 某次训练的结果
'''
After 0 training step(s), the loss result is 0.525303
After 100 training step(s), the loss result is 0.002255
After 200 training step(s), the loss result is 0.000643
...
After 9800 training step(s), the loss result is 0.000001
After 9900 training step(s), the loss result is 0.000001
Mean Square Error is: 0.001200
'''
