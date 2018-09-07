import tensorflow as tf
import tensorflow.contrib.slim as slim

# slim.arg_scope函数可以用于设置默认的参数取值
# 第一个参数是一个函数列表
# 后面的参数都是可选参数，代表默认的参数取值
# 在函数列表中的所有函数将使用默认的参数取值
# 例如：在调用slim.conv2d(net, 320, [1,1])函数时会自动加上stride=1和padding='SAME'的参数
# 如果在函数调用时指定了stride，那么默认的stride不会被调用
# 这进一步减少了代码冗余
with slim.arg_scope([slim.conv2d, slim.max_pool2d, slim.avg_pool2d], stride=1, padding='SAME'):

    # 这里省略了Inception-v3其他模块的代码，并假设net代表上一次的输出结果
    net = tf.placeholder(tf.float32, [None, 32, 32, 1], 'x-input')

    # 为一个Inception模块声明一个统一的变量命名空间
    with tf.variable_scope('Mixed_7c'):

        # 为Inception模块中每一条路径声明一个命名空间
        with tf.variable_scope('Branch_0'):
            branch_0 = slim.conv2d(net, 320, [1, 1], scope='Conv2d_0a_1x1')

        # Inception模块的第二条路径，这条计算路径上的结构本身也是一个Inception结构
        with tf.variable_scope('Branch_1'):
            branch_1 = slim.conv2d(net, 384, [1, 1], scope='Conv2d_0a_1x1')
            branch_1 = tf.concat(3, [
                slim.conv2d(branch_1, 384, [1, 3], scope='Conv2d_0b_1x3'),
                slim.conv2d(branch_1, 384, [3, 1], scope='Conv2d_0c_3x1')
            ])

        # Inception模块的第三条路径，这条计算路径上的结构本身也是一个Inception结构
        with tf.variable_scope('Branch_2'):
            branch_2 = slim.conv2d(net, 448, [1, 1], scope='Conv2d_0a_1x1')
            branch_2 = slim.conv2d(branch_2, 384, [3, 3], scope='Conv2d_0b_3x3')
            branch_2 = tf.concat(3, [
                slim.conv2d(branch_2, 384, [1, 3], scope='Conv2d_0c_1x3'),
                slim.conv2d(branch_2, 384, [3, 1], scope='Conv2d_0d_3x1')
            ])

        # Inception模块的第四条路径，这条计算路径上的结构本身也是一个Inception结构
        with tf.variable_scope('Branch_3'):
            branch_3 = slim.avg_pool2d(net, [3, 3], scope='AvgPool_0a_3x3')
            branch_3 = slim.conv2d(branch_3, 192, [1, 1], scope='Conv2d_0b_1x1')

    # 当前Inception模块的最后输出是由上面4个计算结果拼接得到的
    net = tf.concat(3, [branch_0, branch_1, branch_2, branch_3])
