import tensorflow as tf

v1 = tf.Variable(tf.constant(1.0, shape=[1]), name="v1")
v2 = tf.Variable(tf.constant(2.0, shape=[1]), name="v2")
result = v1 + v2

saver = tf.train.Saver()
# 通过export_meta_graph函数导出TensorFlow计算图的元图，并保存为JSON格式
saver.export_meta_graph("/PycharmProjects/TFDemo/data/model/549/model.ckpt.meta.json", as_text=True)
