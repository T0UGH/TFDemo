import codecs
import collections
from operator import itemgetter

# 训练集数据文件
RAW_DATA = "/PycharmProjects/TFDemo/data/PTB/data/ptb.train.txt"
# 输出的词汇表文件
VOCAB_OUTPUT = "/PycharmProjects/TFDemo/data/PTB/generate/ptb.vocab"

# 统计单词出现的频率
counter = collections.Counter()
with codecs.open(RAW_DATA, 'r', 'utf-8') as file:
    for line in file:
        for word in line.strip().split():
            counter[word] += 1

# 按照词频顺序对单词进行排序
sorted_word_to_cnt = sorted(counter.items(), key=itemgetter(1), reverse=True)
sorted_words = [x[0] for x in sorted_word_to_cnt]

# 加入句子结束符<eos>
sorted_words = ["<eos>"] + sorted_words

# 将排序好的单词写入输出文件，其中行号就代表了单词的编号
with codecs.open(VOCAB_OUTPUT, 'w', 'utf-8') as file_output:
    for word in sorted_words:
        file_output.write(word + '\n')
