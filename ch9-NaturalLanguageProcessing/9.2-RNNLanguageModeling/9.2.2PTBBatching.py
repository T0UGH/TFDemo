import numpy as np

TRAIN_DATA = "/PycharmProjects/TFDemo/data/PTB/generate/ptb.train"
TRAIN_BATCH_SIZE = 20
TRAIN_NUM_STEP = 35


# 从文件中读取数据，并返回包含单词编号的数组
def read_data(file_path):
    with open(file_path, 'r') as fin:
        # 将整个文档读入一个长字符串
        id_string = ' '.join([line.strip() for line in fin.readlines()])
    id_list = [int(w) for w in id_string.split()]
    return id_list


def make_batches(id_list, batch_size, num_step):
    # 计算总的batch数量，每个batch包含的单词数量是batch_size * num_steps
    num_batches = (len(id_list) -1) // (batch_size * num_step)

    # 将数据整理为一个维度为[batch_size, num_batches * num_step]的二维数组
    data = np.array(id_list[:num_batches * batch_size * num_step])
    data = np.reshape(data, [batch_size, num_batches * num_step])

    # 沿着第二个维度将数据切分为num_batches个batch，存入一个数组中
    data_batches = np.split(data, num_batches, axis=1)

    # 重复上述操作，但是每个位置向右移动一位
    # 作为输入数据的标记值，也就是RNN每一步输入所需要预测的下一个单词
    label = np.array(id_list[1: num_batches * batch_size * num_step + 1])
    label = np.reshape(label, [batch_size, num_batches * num_step])
    label_batches = np.split(label, num_batches, axis=1)

    # 返回一个长度为num_batches的数组，其中每一项包括一个data矩阵和一个label矩阵
    return list(zip(data_batches, label_batches))


def main():
    train_batches = make_batches(read_data(TRAIN_DATA), TRAIN_BATCH_SIZE, TRAIN_NUM_STEP)
    print(train_batches)


if __name__ == '__main__':
    main()
