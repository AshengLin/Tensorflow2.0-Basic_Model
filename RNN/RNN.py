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

tokenizer.fit_on_texts(separation_data)
print(tokenizer.index_word)

# word embedding

# Recurrent

# NN