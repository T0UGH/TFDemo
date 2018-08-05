import tensorflow as tf

weights = tf.constant([[1.0, 2.0], [-3.0, 4.0]], name="const_mat", dtype=tf.float32)

# 返回一个函数，这个函数可以计算一个给定参数的L1正则化项的值
regularizer = tf.contrib.layers.l1_regularizer(0.5)
y = regularizer(weights)

with tf.Session() as sess:
    print(sess.run(y))
