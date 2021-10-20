import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import numpy as np
import jieba
from gensim.models import Word2Vec
import os

data_waimai = pd.read_csv('https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/waimai_10k/waimai_10k.csv')
#  jieba.lcut(data_waimai.review[0])  # 詞切割
data_waimai['text'] = data_waimai.review.apply(jieba.lcut)
w2v = Word2Vec(data_waimai.text, vector_size=250, sg=1, min_count=1)  # 出現一次就要加入

# print(w2v.wv.index_to_key)
# print(w2v.wv.key_to_index)
# print([1 + w2v.wv.key_to_index[sen] for sen in data_waimai.text[0]])  # 加1為了之後padding的0

# 所有Vector需要往後推
embedding_matrix = w2v.wv.vectors
# print(embedding_matrix.shape)
embedding_matrix = np.vstack((np.array(np.zeros(250)), embedding_matrix))  # 補一個padding 0用的vector
# print(embedding_matrix.shape)

x_all = np.zeros([len(data_waimai.text), 30], dtype='float64')
for i in range(len(data_waimai.text)):
    for j in range(min(len(data_waimai.text[i]), 30)):  # 最多30就好 統一長度
        x_all[i, j] = 1 + w2v.wv.key_to_index[data_waimai.text[i][j]]  # 加1為了之後padding的0
y_all = data_waimai.label

x_train, x_test = x_all[:10000], x_all[10000:]
y_train, y_teat = y_all[:10000], y_all[10000:]


lstm_model = tf.keras.Sequential(name='LSTM')
lstm_model.add(layers.Embedding(len(w2v.wv) + 1, 250))  # 記得1要加
lstm_model.add(layers.LSTM(64))
lstm_model.add(layers.Dense(2, activation='softmax'))

if os.path.exists('lstm_model.png'):
    os.remove('lstm_model.png')
tf.keras.utils.plot_model(lstm_model, show_shapes=True, to_file='lstm_model.png')

lstm_model.compile(optimizer='Adam',
                   loss=keras.losses.sparse_categorical_crossentropy,
                   metrics=['accuracy'])

lstm_model.fit(x_train, y_train.values, epochs=20)
predict_y = lstm_model.predict_classes(x_test)
lstm_model.evaluate(x_test, y_teat.values)
