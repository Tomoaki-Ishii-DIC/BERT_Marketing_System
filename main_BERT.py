import sys
import os
import re
import codecs
import json

from transformers import BertJapaneseTokenizer

import pandas as pd

from preprocessing import preprocessing
from preprocessing import make_datasets
from preprocessing import change_config

from keras.utils import np_utils
from keras import utils
import numpy as np

import tensorflow_addons as tfa
import keras
from keras_bert import load_trained_model_from_checkpoint
from keras_bert.layers import MaskedGlobalMaxPool1D
from keras import Input, Model
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Flatten, GlobalMaxPooling1D
from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard


def create_model():
    """
    BERTのモデルを作成する関数

    Parameters
    ----------------

    """
    config_file = os.path.join('./BERT-base_mecab-ipadic-bpe-32k', 'config.json')
    checkpoint_file = os.path.join('./BERT-base_mecab-ipadic-bpe-32k', 'model.ckpt')

    pre_trained_model = load_trained_model_from_checkpoint(config_file,
                                                            checkpoint_file,
                                                            training=True,
                                                            seq_len=SEQ_LEN)
    pre_trained_output  = pre_trained_model.get_layer(name='NSP-Dense').output
    model_output = Dense(2, activation='softmax')(pre_trained_output)

    model  = Model(inputs=[pre_trained_model.input[0],
                    pre_trained_model.input[1]],
                    outputs=model_output)

    model.compile(loss='categorical_crossentropy',
                    #勾配消失防止のためAdamWarmupを使用する
                    optimizer = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.001),
                    metrics=['mae', 'mse', 'acc'])

    return model

# データセットの作成
df_train_features, df_test_features, df_train_labels, df_test_labels = make_datasets.make_ds()

# BERTのトークン最大値を取得
json_path = "./BERT-base_mecab-ipadic-bpe-32k/config.json"
with open(json_path) as f:
    data = json.load(f)
maxlen = data["max_position_embeddings"]


##### ラベル側の処理 #####

# クラス数（何種類に分類するか）ネガポジなら２ {0: 'positive', 1: 'negative'}
class_count = 2

# labelをワンホット表現に変換
#train_dum = pd.get_dummies(df_train_labels)
#test_dum = pd.get_dummies(df_test_labels)
#train_labels = np.array(train_dum[['positive', 'negative']])
#test_labels = np.array(test_dum[['positive', 'negative']])

# labelをワンホット表現に変換
df_train_num = df_train_labels.replace('positive', 0).replace('negative', 1)
train_ndarray_labels = df_train_num.astype(int)
train_labels = np.identity(2)[train_ndarray_labels].astype(int)

df_test_num = df_test_labels.replace('positive', 0).replace('negative', 1)
test_ndarray_labels = df_test_num.astype(int)
test_labels = np.identity(2)[test_ndarray_labels].astype(int)

##### 特徴量側の処理 #####
tknz = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')

train_features = []
test_features = []
for feature in df_train_features:
    # ID化
    train_features.append(preprocessing.get_indice(feature, maxlen, tknz))
train_features = np.array(train_features)

# shape(len(train_features), maxlen)のゼロの行列作成
train_segments = np.zeros((len(train_features), maxlen), dtype = np.float32)

for feature in df_test_features:
    # ID化
    test_features.append(preprocessing.get_indice(feature, maxlen, tknz))
test_features = np.array(test_features)

# shape(len(test_features), maxlen)のゼロの行列作成
test_segments = np.zeros((len(test_features), maxlen), dtype = np.float32)

# パラメータ
SEQ_LEN = maxlen
#BATCH_SIZE = 8
BATCH_SIZE = 4 #メモリ不足によるResourceExhaustedError対策　
#BERT_DIM = 768
LR = 1e-4

# 学習回数
EPOCH = 30

# 設定の変更 BERTのサイズを減らしてメモリを節約
#change_config.set_config(SEQ_LEN)

# モデルの作成
model = create_model()
model.summary()

# コールバック用　チェックポイント保存用
checkpoint_path = './models/finetuning_checkpoint'

check_point = ModelCheckpoint(monitor='val_acc',
                                mode='max',
                                filepath=checkpoint_path,
                                save_best_only=True)

early_stopping = EarlyStopping(monitor = "val_loss",
                                min_delta=0.001,
                                patience=5,
                                verbose=1,
                                mode="min",
                                restore_best_weights=False)

reduce_lr = ReduceLROnPlateau(monitor="val_loss",
                                factor=0.1,
                                patience=2,
                                verbose=1,
                                mode="min",
                                min_delta=0.0001)

# 学習
history = model.fit([train_features, train_segments],
                      train_labels,
                      epochs = EPOCH,
                      batch_size = BATCH_SIZE,
                      validation_data=([test_features, test_segments], test_labels),
                      shuffle=False,
                      verbose = 1,
                      callbacks = [check_point, early_stopping, reduce_lr])

# モデルの保存
model.save('./models/saved_model_BERT')
