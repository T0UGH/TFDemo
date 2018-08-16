import tensorflow as tf

# 声明两个变量并计算它们的和
v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

init_op = tf.global_variables_initializer()

# 声明tf.train.Saver类用来保存模型
saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(init_op)
    print(sess.run(result))
    # 将模型保存到D:PycharmProjects/TFDemo/data/model/540/model.ckpt
    saver.save(sess, "/PycharmProjects/TFDemo/data/model/540/model.ckpt")
