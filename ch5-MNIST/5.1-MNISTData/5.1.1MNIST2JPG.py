import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
tf.reset_default_graph()
im = mnist.test.images[4].reshape((28, 28))                         # 读取的格式为Ndarry
img = Image.fromarray(im*255)                                       # Image和Ndarray互相转换
img = img.convert('RGB')                                            # jpg可以是RGB模式，也可以是CMYK模式
img.save(r'D:/PycharmProjects/TFDemo/data/MNIST/img/2.jpg')         # 保存
# plt.imshow(im,cmap="gray")
# plt.show()                                                        # 显示
