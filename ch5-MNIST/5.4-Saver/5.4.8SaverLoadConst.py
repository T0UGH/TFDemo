import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename = "/PycharmProjects/TFDemo/data/model/547/combined_model.pb"

    # 读取保存的模型文件，将文件解析为对应的GraphDef Protocal Buffer
    with gfile.FastGFile(model_filename, 'rb') as file:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(file.read())

    # 将graph_def中保存的图加载到当前的图中。
    # return_elements给出了返回的张量的名称。在保存时给出的是计算节点的名称，所以为"add"，在加载时给出的是张量的名称，所以为add:0
    result = tf.import_graph_def(graph_def, return_elements=["add:0"])
    # 输出[3.0]　　
    print(sess.run(result))
