import glob
import os.path
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import  gfile
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.python.slim.nets.inception_v3 as inception_v3

# 处理好的数据文件
INPUT_DATA = '/PycharmProjects/TFDemo/data/flower_processed_data.npy'
# 保存训练好的模型的路径
TRAIN_FILE = '/PycharmProjects/TFDemo/data/save_model'
# 谷歌提供的训练好的模型文件地址
CKPT_FILE = '/PycharmProjects/TFDemo/data/inception_v3.ckpt'

# 定义训练中使用的参数
LEARNING_RATE = 0.0001
STEPS = 300
BATCH = 32
N_CLASSES = 5

CHECKPOINT_EXCLUDE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'

TRAINABLE_SCOPES = 'InceptionV3/Logits,InceptionV3/AuxLogits'

def get_tuned_variables():
    exclusions = []
