from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Embedding
from tensorflow.keras.callbacks import LambdaCallback
import numpy as np
import random
import sys
import pickle


poetry = load_model('poetry.h5')  # 读取模型
with open('dictionary.pkl', 'rb') as fw:
    try:
        data = pickle.load(fw)
    except Exception as e:
        print(e)
char2id = data[0]
id2char = data[1]


# 预测时，采样函数
def sample(preds, diversity=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds + 1e-10) / diversity
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# 文本生成
def on_epoch_end(sen, maxlen):
    for diversity in [0.2, 0.5, 0.8, 1.0]:
        print('----- diversity:', diversity)
        # sentence = sentences[index][:maxlen]   #随机找一串文本作为起始语句
        sentence = sen
        print('----- Generating with seed: ' + sentence)
        sys.stdout.write(sentence)

        for i in range(42):     # 预测42个字
            x_pred = np.zeros((1, maxlen))
            for t, char in enumerate(sentence):
                x_pred[0, t] = char2id[char]

            preds = poetry.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = id2char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            # with open('haha.txt', 'w') as f:
            #     f.write(next_char+'\n')
            sys.stdout.flush()


on_epoch_end('潮起又潮落,', 6)
