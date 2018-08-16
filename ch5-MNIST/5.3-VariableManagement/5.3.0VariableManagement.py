import tensorflow as tf

v1 = tf.get_variable("v", [1])
print(v1.name)          # 输出v:0, "v"为变量名称, ":0"表示此变量是生成这个变量的运算的第一个输出

with tf.variable_scope("foo"):
    v2 = tf.get_variable("v", [1])
    print(v2.name)      # 输出foo/v:0, 因为在名字空间中，所以变量名称前会加入名字空间的名称

with tf.variable_scope("foo"):
    with tf.variable_scope("bar"):
        v3 = tf.get_variable("v", [1])
        print(v3.name)  # 输出foo/bar/v:0,
    v4 = tf.get_variable("v1", [1])
    print(v4.name)      # 输出foo/v1:0

# 创建一个名称为空的名字空间，并设置reuse=True
with tf.variable_scope("", reuse=True):

    # 可以直接通过带名字空间名称的变量名来获取其他名字空间下的变量
    # 比如这里直接通过指定"foo/bar/v"，来获得名字空间foo/bar下的v变量
    v5 = tf.get_variable("foo/bar/v", [1])
    print(v5 == v3)     # 输出True

    v6 = tf.get_variable("foo/v", [1])
    print(v6 == v2)     # 输出True

