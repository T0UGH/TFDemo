import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")

# 在没有声明滑动平均模型时，只有一个变量v
for variable in tf.global_variables():
    # 输出"v:0"
    print(variable.name)

# 声明滑动平均模型
ema = tf.train.ExponentialMovingAverage(0.99)
# 为`tf.global_variables()`中每个变量生成一个影子变量
maintain_average_op = ema.apply(tf.global_variables())

for variable in tf.global_variables():
    print(variable.name)

saver = tf.train.Saver()

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(tf.assign(v, 10))
    sess.run(maintain_average_op)
    saver.save(sess, "/PycharmProjects/TFDemo/data/model/544/model.ckpt")
    print(sess.run([v, ema.average(v)]))
