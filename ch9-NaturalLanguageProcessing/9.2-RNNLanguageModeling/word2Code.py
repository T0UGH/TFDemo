import codecs

RAW_DATA = "/PycharmProjects/TFDemo/data/PTB/data/ptb.train.txt"
VOCAB = "/PycharmProjects/TFDemo/data/PTB/generate/ptb.vocab"
OUTPUT_DATA = "/PycharmProjects/TFDemo/data/PTB/generate/ptb.train"

# 读取词汇表，并建立词汇到单词编号的映射
with codecs.open(VOCAB, "r", "utf-8") as f_vocab:
    vocab = [w.strip() for w in f_vocab.readlines()]
word_to_code = {k: v for (k, v) in zip(vocab, range(len(vocab)))}

# 打开输入与输出文件
fin = codecs.open(RAW_DATA, "r", "utf-8")
fout = codecs.open(OUTPUT_DATA, "w", "utf-8")

# 根据词汇映射进行转码
for line in fin:
    words = line.strip().split() + ["<eos>"]
    out_line = ' '.join([str(word_to_code[w]) for w in words]) + '\n'
    fout.write(out_line)

# 关闭文件
fin.close()
fout.close()
