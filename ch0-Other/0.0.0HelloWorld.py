import tensorflow as tf
a = tf.constant([1.0, 2.0],name="a")
b = tf.constant([2.0, 3.0],name="b")
result = a + b
print(result)
config = tf.ConfigProto(
    allow_soft_placement=True,
    log_device_placement=True
)
with tf.Session(config=config) as sess:
    with sess.as_default():
        print(result.eval())
