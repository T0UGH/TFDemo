import tensorflow as tf

# 使用和保存模型代码中一样的方式来声明变量
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(1.0, shape=[1]), name="v2")
result = v1 + v2

saver = tf.train.Saver()

with tf.Session() as sess:
    # 通过saver.restore加载v1和v2的具体取值，不需要初始化过程
    saver.restore(sess, "/PycharmProjects/TFDemo/data/model/540/model.ckpt")
    print(sess.run(result))
