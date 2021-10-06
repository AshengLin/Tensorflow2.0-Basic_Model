import tensorflow as tf
import pandas as pd
import jieba
from gensim.models import Word2Vec

data_waimai = pd.read_csv('https://raw.githubusercontent.com/SophonPlus/ChineseNlpCorpus/master/datasets/waimai_10k/waimai_10k.csv')
#  jieba.lcut(data_waimai.review[0])  # 詞切割
data_waimai['text'] = data_waimai.review.apply(jieba.lcut)
w2v = Word2Vec(data_waimai.text)
