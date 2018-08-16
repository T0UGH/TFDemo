import tensorflow as tf

v = tf.Variable(0, dtype=tf.float32, name="v")
ema = tf.train.ExponentialMovingAverage(.99)

# 通过使用variables_to_restore()函数可以直接生成5.4.5代码中提供的字典
# {"v/ExponentialMovingAverage": v}
# 输出：{'v/ExponentialMovingAverage': <tf.Variable 'v:0' shape=() dtype=float32_ref>}
print(ema.variables_to_restore())

saver = tf.train.Saver(ema.variables_to_restore())

with tf.Session() as sess:
    saver.restore(sess, "/PycharmProjects/TFDemo/data/model/544/model.ckpt")
    print(sess.run(v))
