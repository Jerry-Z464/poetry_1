from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.callbacks import LambdaCallback
import numpy as np
import random
import sys
import pickle


# 统计五言律诗的数量
sentences = []
with open('./poetry_data/poetry1.txt', 'r', encoding='utf8') as fr:
    lines = fr.readlines()
    for line in lines:
        line = line.strip()
        sentences.append(line)
print('共%d首诗' % len(sentences))
print('前5首诗:')
for i in sentences[:5]:
    print(i)

# 构建字典，并为其每个字一个序号
chars = {}
for sentence in sentences:
    for c in sentence:
        chars[c] = chars.get(c, 0) + 1
chars = sorted(chars.items(), key=lambda x: x[1], reverse=True)
chars = [char[0] for char in chars]
vocab_size = len(chars)
print('共%d个字' % vocab_size, chars[:10])
# 字典的转换关系
char2id = {c: i for i, c in enumerate(chars)}
id2char = {i: c for i, c in enumerate(chars)}
with open('dictionary.pkl', 'wb') as fw:
    try:
        pickle.dump([char2id, id2char], fw)
    except Exception as e:
        print(e)
print(char2id)
print(id2char)

#设定参数
maxlen = 6    #考虑5个字符序列
embed_size = 256        #词向量长度
hidden_size = 256        #隐藏状态长度
vocab_size = len(chars)  #词汇表长度（输出空间大小）
batch_size = 64
epochs = 15


# 构建数据集，耗时长
X_data = []
Y_data = []
for sentence in sentences:
    for i in range(0, len(sentence) - maxlen):
        X_data.append([char2id[c] for c in sentence[i: i + maxlen]])
        y = np.zeros(vocab_size, dtype=np.bool)
        y[char2id[sentence[i+maxlen]]] = 1
        Y_data.append(y)


X_data = np.array(X_data)
Y_data = np.array(Y_data)
print(X_data[:8])
print(Y_data[:8])
# print(X_data.shape, Y_data.shape)
