import tensorflow as tf

# `tf.train.NewCheckpointReader`可以读取checkpoint文件中保存的所有变量
reader = tf.train.NewCheckpointReader('/PycharmProjects/TFDemo/data/model/540/model.ckpt')

# 获取所有变量列表，这是一个从变量名到变量维度的字典
global_variables = reader.get_variable_to_shape_map()
for variable_name in global_variables:
    print(variable_name, global_variables[variable_name])

print("Value for variable v1 is ",reader.get_tensor("v1"))
