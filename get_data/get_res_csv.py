# -*- coding: UTF-8 -*-
import os
import re
import pandas as pd
import numpy as np
# 繁体字转简文
from langconv import *


def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence

# 将所有文本和标题汇总成一个txt
def get_text(base_path, out_path):
    data = pd.read_csv(base_path, usecols=[2, 3], encoding='UTF-8')
    data.columns = ["paragraphs", "title"]
    f2 = open(out_path, 'w', encoding='UTF-8')
    for i in range(len(data)):
        try:
            title = data["title"][i]
            title = Converter('zh-hans').convert(title)
            paraagraphs = data["paragraphs"][i]
            # paraagraphs = ''.join(re.findall(u'[\u4e00-\u9fff]+', paraagraphs))
            paraagraphs = Converter('zh-hans').convert(paraagraphs)
            # f2.write(title+' ')
            # f2.write(str(np.squeeze(data.iloc[i, [0]].values)) + ',')
            if len(paraagraphs)==48:
                f2.write(paraagraphs + '\n')
                continue
        except:
            print(data.iloc[i, [0]].values)

    f2.close()

if __name__ == '__main__':
    base_path = "../data/data_poetry10000.csv"
    out_path = "../poetry_data/poetry10.txt"
    get_text(base_path, out_path)


