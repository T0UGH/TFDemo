import tensorflow as tf

# 定义一个变量用来计算滑动平均，这个变量的初始值为0
v1 = tf.Variable(0, dtype=tf.float32)
# 定义step用来模拟神经网络迭代的轮数，可以用来动态控制衰减率
step = tf.Variable(0, trainable=False)

# 定义一个滑动平均的类，初始化时给定了衰减率(0.99)和控制衰减率的变量step
ema = tf.train.ExponentialMovingAverage(0.99, step)

# 定义一个更新变量滑动平均的操作，这里需要给定一个列表，每次执行这个操作时，这个列表中的变量都会被更新
maintain_averages_op = ema.apply([v1])

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    # 通过ema.average(v1)来获得滑动平均之后变量的取值
    print(sess.run([v1, ema.average(v1)]))

    # 更新变量v1的值到5
    sess.run(tf.assign(v1, 5))
    # 使用maintain_averages_op操作来更新v1的滑动平均值
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

    # 更新变量step的值到10000
    sess.run(tf.assign(step, 10000))
    # 更新变量v1的值到10
    sess.run(tf.assign(v1, 10))
    # 使用maintain_averages_op操作来更新v1的滑动平均值
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))

    # 再次更新滑动平均值
    sess.run(maintain_averages_op)
    print(sess.run([v1, ema.average(v1)]))
