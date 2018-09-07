import tensorflow.contrib.slim as slim
import tensorflow as tf

# 直接使用TensorFlow原始API实现卷积层
with tf.variable_scope('scope-name'):
    weights = tf.get_variable('weights', [32, 32, 1, 5], initializer=tf.truncated_normal_initializer(stddev=0.1))
    biases = tf.get_variable('biases', [5], initializer=tf.constant_initializer(0.0 ))
    # 这里用input代表上一层的输入，并不是实际存在的变量
    conv = tf.nn.conv2d(input, weights, strides=[1, 1, 1, 1], padding="SAME")
    relu = tf.nn.relu(tf.nn.bias_add(conv, biases))

# 使用slim实现卷积层
# slim.conv2d函数有3个必填参数
# 第一个参数为输入矩阵
# 第二个参数为当前卷积层过滤器的深度
# 第三个参数为过滤器的尺寸
# 可选参数有过滤器移动的步长，是否使用全0填充，激活函数的选择，变量的命名空间等
net = slim.conv2d(input, 32, [3, 3])

