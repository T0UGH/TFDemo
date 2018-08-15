import tensorflow as tf

# 在名字为foo的命名空间内创建名字为v的变量
with tf.variable_scope("foo"):
    v = tf.get_variable(
        "v", [1], initializer=tf.constant_initializer(1.0)
    )

# 因为在命名空间foo中已经存在名字为v的变量，所以以下代码将会报错
with tf.variable_scope("foo"):
    v = tf.get_variable("v", [1])

# 在生成上下文管理器时，将参数reuse设置为True。这样tf.get_variable()函数将直接获得已经声明的变量
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v", [1])
    print(v == v1)      # 输出为True，代表v,v1表示的是相同的TensorFlow中变量

# 将参数reuse设置为True时，tf.variable_scope将只能获得已经创建过的变量，因为在命名空间bar中还没有创建变量v，所以下面会报错
with tf.variable_scope("bar", reuse=True):
    v3 = tf.get_variable("v", [1])
