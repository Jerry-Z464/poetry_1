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

# 构建字典，并为其每个字一个序号
chars = {}
for sentence in sentences:
    for c in sentence:
        chars[c] = chars.get(c, 0) + 1
chars = sorted(chars.items(), key=lambda x: x[1], reverse=True)
chars = [char[0] for char in chars]
vocab_size = len(chars)
print('共%d个字' % vocab_size, chars[:20])
# 字典的转换关系
char2id = {c: i for i, c in enumerate(chars)}
id2char = {i: c for i, c in enumerate(chars)}
with open('dictionary.pkl', 'wb') as fw:
    try:
        pickle.dump([char2id, id2char], fw)
    except Exception as e:
        print(e)


#设定参数
maxlen = 6   #考虑5个字符序列
embed_size = 256        #词向量长度
hidden_size = 256        #隐藏状态长度
vocab_size = len(chars)  #词汇表长度（输出空间大小）
batch_size = 64
epochs = 5

#构建数据集，耗时长
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
print(X_data.shape, Y_data.shape)

#构建LSTM网络，包含嵌入层、LSTM层，以及全连接层
#可以引入dropout层以避免过拟合
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=maxlen))  # 词向量维度为256
model.add(LSTM(hidden_size, input_shape=(maxlen, embed_size)))  # 构建LSTM模型
model.add(Dense(vocab_size, activation='softmax'))  # 输出层，输出词汇表里有的字
model.compile(loss='categorical_crossentropy', optimizer='adam')

#预测时，采样函数
def sample(preds, diversity=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / diversity
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

#训练完后，进行文本生成
def on_epoch_end(epoch, logs):
    print('-' * 30)
    print('Epoch', epoch)

    index = random.randint(0, len(sentences))
    for diversity in [0.2, 0.5, 0.8, 1.0]:
        print('----- diversity:', diversity)
        sentence = sentences[index][:maxlen]   #随机找一串文本作为起始语句
        print('----- Generating with seed: ' + sentence)
        sys.stdout.write(sentence)

        for i in range(42):     #预测42个字
            x_pred = np.zeros((1, maxlen))
            for t, char in enumerate(sentence):
                x_pred[0, t] = char2id[char]

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = id2char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()


model.fit(X_data, Y_data, batch_size=batch_size, epochs=epochs, callbacks=[LambdaCallback(on_epoch_end=on_epoch_end)])
model.save('poetry.h5')  # 保存模型
