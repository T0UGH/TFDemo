import tensorflow as tf

# 这里声明的变量名称与已经保存的模型中的变量的名称不同
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="other-v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="other-v2")
result = v1 + v2

# 若直接使用tf.train.Saver()加载模型会报变量找不到的错误
# 使用一个字典来重命名变量可以加载原来的模型
saver = tf.train.Saver({"v1": v1, "v2": v2})

with tf.Session() as sess:
    saver.restore(sess, "/PycharmProjects/TFDemo/data/model/540/model.ckpt")
    print(sess.run(result))

