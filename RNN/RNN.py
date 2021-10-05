import os
import numpy as np
import pandas as pd
import re
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


data_waimai = pd.read_csv('https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/waimai_10k/waimai_10k.csv')
pattern = re.compile('.{1}')
# print(pattern.findall(data_waimai.review[0]))
separation_data = [pattern.findall(s) for s in data_waimai.review]
tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
tokenizer.fit_on_texts(separation_data)  # encoding
print(tokenizer.index_word)
convert_text = tokenizer.texts_to_sequences(separation_data)  # word to token
convert_text = tf.keras.preprocessing.sequence.pad_sequences(
    convert_text,
    padding='post',  # 補在後面 可選 'pre','post'
    truncating='post',  # 超出maxlen的刪除 可選 'pre','post'
    maxlen=30
)  # padding

# word embedding
numWords = len(tokenizer.index_word)
embedding_dim = 250

rnn_model = tf.keras.Sequential()
rnn_model.add(layers.Embedding(numWords+1, embedding_dim))  # 1 for zero, padding

# RNN layers
rnn_model.add(layers.SimpleRNN(64))

# Classification layers
rnn_model.add(layers.Dense(2, activation='softmax'))

rnn_model.compile(optimizer='adam',
                  loss=tf.losses.sparse_categorical_crossentropy,
                  metrics=['accuracy'])

rnn_model.summary()

if os.path.exists('model.png'):
    os.remove('model.png')
tf.keras.utils.plot_model(rnn_model, show_shapes=True, to_file='model.png')

rnn_model.fit(convert_text, data_waimai.label.values, epochs=10)
