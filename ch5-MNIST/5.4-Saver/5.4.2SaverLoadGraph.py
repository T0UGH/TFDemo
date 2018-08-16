import tensorflow as tf

# 直接加载持久化的图
saver = tf.train.import_meta_graph("/PycharmProjects/TFDemo/data/model/540/model.ckpt.meta")

# 通过名字获得result变量
result = tf.get_default_graph().get_tensor_by_name("add:0")

with tf.Session() as sess:

    # 通过saver.restore加载v1和v2的具体取值，不需要初始化过程
    saver.restore(sess, "/PycharmProjects/TFDemo/data/model/540/model.ckpt")
    # 输出[3.]
    print(sess.run(result))
