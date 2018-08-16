import tensorflow as tf
from numpy.random import RandomState


# 返回一层神经网络边上的权重，并将这个权重的L2正则化损失加入名称为'losses'的集合中
def get_weight(shape, lamb):
    # 生成一个变量
    var = tf.Variable(tf.random_normal(shape), dtype=tf.float32)

    # 将新生成变量的L2正则化损失项加入集合
    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(lamb)(var))

    # 返回这个变量
    return var


def print_weights(sess):
    variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    print()
    for var in variables:
        print()
        print(var)
        print(sess.run(var))


# 定义输入数据
x = tf.placeholder(tf.float32, (None, 2))
y_ = tf.placeholder(tf.float32, (None, 1))

# 定义batch的大小
batch_size = 8

# 定义每层网络中神经元的个数
layer_dimension = [2, 10, 10, 10, 1]
# 神经网络的层数
n_layers = len(layer_dimension)

# 当前层的神经元
cur_layer = x
# 当前层的神经元个数
in_dimension = layer_dimension[0]

# 通过一个循环来生成5层全连接的神经网络
for i in range(1, n_layers):
    # 下一层的节点个数
    out_dimension = layer_dimension[i]
    # 根据这一层和下一层的节点数来生成当前层的权重，并将L2正则添加到集合
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))

    # 使用RuLU激活函数计算下一层的节点
    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)
    # 进入下一层之前将下一层的节点数更新为当前层的节点数
    in_dimension = layer_dimension[i]

# 使用均方误差作为损失函数
mes_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

# 将均方误差添加到集合
tf.add_to_collection('losses', mes_loss)

# 将集合中所有元素相加，得到最后的损失函数（包括MSE和正则）
loss = tf.add_n(tf.get_collection('losses'))

global_step = tf.Variable(0)

# 使用指数衰减来设置学习率
learning_rate = tf.train.exponential_decay(0.1, global_step, 100, 0.96, staircase=True)

# 定义优化过程
learning_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

# 使用Numpy生成随机数
rdm = RandomState(1)

# 定义数据集的大小
dataset_size = 128

# 使用前面的随机数来生成输入数据X和Y
X = rdm.rand(dataset_size, 2)
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

# 开启会话开始运算
with tf.Session() as sess:
    # 初始化所有变量
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 打印所有参数
    print_weights(sess)

    # 设置训练的轮数
    STEPS = 5000

    # 开启迭代的训练过程
    for i in range(STEPS):

        # 设置这一组batch的开始值和结束值
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)

        # 训练神经网络并更新参数
        sess.run(learning_step, feed_dict={
            x: X[start:end],
            y_: Y[start:end]
        })

    # 打印所有参数
    print_weights(sess)
