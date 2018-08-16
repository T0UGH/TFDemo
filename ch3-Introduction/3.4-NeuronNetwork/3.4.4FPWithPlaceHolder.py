import tensorflow as tf
w1 = tf.Variable(tf.random_normal((2, 3), stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal((3, 1), stddev=1, seed=1))
x = tf.placeholder(tf.float32,shape=(3, 2), name="input")
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)
config=tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=True
)
with tf.Session(config=config) as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(y, feed_dict={x: [[0.7, 0.9], [0.1, 0.4], [0.5, 0.8]]}))
