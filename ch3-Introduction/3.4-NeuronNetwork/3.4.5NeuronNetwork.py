import tensorflow as tf
from numpy.random import RandomState

# 定义训练数据batch的大小
batch_size = 8

# 定义神经网络的参数
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 使用placeholder来定义输入数据
# 其中将shape的一个维度设置为None可以方便使用不同的batch大小
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

# 定义神经网络前向传播的过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数
y = tf.sigmoid(y)
cross_entropy = -tf.reduce_mean(
    y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1-y_) * tf.log(tf.clip_by_value(1-y, 1e-10, 1.0))
)

# 定义优化过程
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

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

    # 训练开始之前打印参数
    print(sess.run(w1))
    print(sess.run(w2))

    # 设置训练的轮数
    STEPS = 5000

    # 开启迭代的训练过程
    for i in range(STEPS):

        # 设置这一组batch的开始值和结束值
        start = (i * batch_size) % dataset_size
        end = min(start+batch_size, dataset_size)

        # 训练神经网络并更新参数
        sess.run(train_step, feed_dict={
            x: X[start:end],
            y_: Y[start:end]
        })

        # 每隔1000轮算一遍交叉熵，以观察训练结果
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

    # 训练结束打印参数
    print(sess.run(w1))
    print(sess.run(w2))
