import pandas as pd
import numpy as np
import json


data_poetry1000 = pd.read_json('poet.tang.41000.json', orient='values', lines=False)
data_poetry2000 = pd.read_json('poet.tang.42000.json', orient='values', lines=False)
data_poetry3000 = pd.read_json('poet.tang.43000.json', orient='values', lines=False)
data_poetry4000 = pd.read_json('poet.tang.44000.json', orient='values', lines=False)
data_poetry5000 = pd.read_json('poet.tang.45000.json', orient='values', lines=False)
data_poetry6000 = pd.read_json('poet.tang.46000.json', orient='values', lines=False)
data_poetry7000 = pd.read_json('poet.tang.47000.json', orient='values', lines=False)
data_poetry8000 = pd.read_json('poet.tang.48000.json', orient='values', lines=False)
data_poetry9000 = pd.read_json('poet.tang.49000.json', orient='values', lines=False)
data_poetry10000 = pd.read_json('poet.tang.50000.json', orient='values', lines=False)
for i in range(len(data_poetry1000['paragraphs'])):
    data_poetry1000['paragraphs'][i] = ''.join(data_poetry1000['paragraphs'][i])
for i in range(len(data_poetry2000['paragraphs'])):
    data_poetry2000['paragraphs'][i] = ''.join(data_poetry2000['paragraphs'][i])
for i in range(len(data_poetry3000['paragraphs'])):
    data_poetry3000['paragraphs'][i] = ''.join(data_poetry3000['paragraphs'][i])
for i in range(len(data_poetry4000['paragraphs'])):
    data_poetry4000['paragraphs'][i] = ''.join(data_poetry4000['paragraphs'][i])
for i in range(len(data_poetry5000['paragraphs'])):
    data_poetry5000['paragraphs'][i] = ''.join(data_poetry5000['paragraphs'][i])
for i in range(len(data_poetry6000['paragraphs'])):
    data_poetry6000['paragraphs'][i] = ''.join(data_poetry6000['paragraphs'][i])
for i in range(len(data_poetry7000['paragraphs'])):
    data_poetry7000['paragraphs'][i] = ''.join(data_poetry7000['paragraphs'][i])
for i in range(len(data_poetry8000['paragraphs'])):
    data_poetry8000['paragraphs'][i] = ''.join(data_poetry8000['paragraphs'][i])
for i in range(len(data_poetry9000['paragraphs'])):
    data_poetry9000['paragraphs'][i] = ''.join(data_poetry9000['paragraphs'][i])
for i in range(len(data_poetry10000['paragraphs'])):
    data_poetry10000['paragraphs'][i] = ''.join(data_poetry10000['paragraphs'][i])

data_poetry1000.to_csv('data_poetry1000.csv')
data_poetry2000.to_csv('data_poetry2000.csv')
data_poetry3000.to_csv('data_poetry3000.csv')
data_poetry4000.to_csv('data_poetry4000.csv')
data_poetry5000.to_csv('data_poetry5000.csv')
data_poetry6000.to_csv('data_poetry6000.csv')
data_poetry7000.to_csv('data_poetry7000.csv')
data_poetry8000.to_csv('data_poetry8000.csv')
data_poetry9000.to_csv('data_poetry9000.csv')
data_poetry10000.to_csv('data_poetry10000.csv')
