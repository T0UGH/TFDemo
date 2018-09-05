import tensorflow as tf

# 通过tf.get_variable的方式创建过滤器的权重变量
# 卷积层的参数个数只和过滤器的尺寸、深度以及当前层节点矩阵的深度有关
# 此处声明的四维矩阵，前两维表示的是过滤器的尺寸，第三维是当前层的深度，第四维是过滤层的深度
filter_weight = tf.get_variable('weights', [5, 5, 3, 16], initializer=tf.truncated_normal_initializer(stddev=0.1))

# 通过tf.get_variable的方式创建过滤器的偏重变量
# 偏置向量的大小等于过滤层的深度
# 原因是卷积层不仅共享权重矩阵也共享偏置向量
biases = tf.get_variable('biases', [16], initializer=tf.constant_initializer(0.1))

# tf.nn.conv2d是用来实现卷积层前向传播的函数
# 第一个参数为当前层的节点矩阵，这个矩阵为4维矩阵，后面三个维度表示一个节点矩阵，前面一个维度表示一个输入batch
# 第二个参数为卷积层的权重
# 第三个参数为不同维度上的步长，是一个长度为4的数组，但第一维和最后一维都是1，因为步长只对矩阵的长和宽有用
# 第四个参数为填充方法，其中"SAME"表示全0填充，"VALID"表示不填充
conv = tf.nn.conv2d(input, filter_weight, strides=[1, 1, 1, 1], padding='SAME')

# tf.nn.bias_add提供了方便的函数给每一个节点加上偏置项
# 这里不能直接使用加法
# 因为矩阵上不同位置上的节点都需要加上同样的偏置项，但是偏置项只有一个数
bias = tf.nn.bias_add(conv, biases)

# 最后使用ReLU激活函数完成去线性化
actived_conv = tf.nn.relu(bias)
