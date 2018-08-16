from tensorflow.examples.tutorials.mnist import input_data

# 载入MNIST数据集，如果指定地址/path/to/MNIST_data下没有已经下载好的数据
# 那么TensorFlow会自动下载数据
mnist = input_data.read_data_sets("/path/to/MNIST_data/", one_hot=True)

# 打印训练集的数据量
print("Training data size:", mnist.train.num_examples)

# 打印验证集的数据量
print("Validating data size:", mnist.validation.num_examples)

# 打印测试集的数据量
print("Testing data size:", mnist.test.num_examples)

# 打印训练集中某一样例数据
# 一张数字图片，它的每个像素点都被放到这个长度为128一维数组中
# 如果一个像素点越接近于1，则颜色越深；越接近于0，则颜色越浅
print("Example training data:", mnist.train.images[0])

# 打印训练集中某一样例数据的标记
# 一个大小为10的一维数组
# 数组中其中一个数字取值为1,其余数字取值为0
print("Example training data label:", mnist.train.labels[0])

# 设置batch的大小
batch_size = 100

# 使用next_batch()方法来获得下一个batch的输入数据
xs, ys = mnist.train.next_batch(batch_size)

print("X shape:", xs.shape)
print("Y shape", ys.shape)
