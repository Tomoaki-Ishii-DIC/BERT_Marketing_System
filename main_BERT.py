import sys
import os
import re
import codecs
import sentencepiece as spm

import pandas as pd

from preprocessing import preprocessing
from preprocessing import make_datasets
from preprocessing import change_config


from keras.utils import np_utils
from keras import utils
import numpy as np


import keras
from keras_bert import load_trained_model_from_checkpoint
from keras_bert.layers import MaskedGlobalMaxPool1D
from keras import Input, Model
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Flatten, GlobalMaxPooling1D
from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.optimizers import Adam
import keras.backend as K #?


def create_model():
    """
    BERTのモデルを作成する関数

    Parameters
    ----------------

    """
    config_file = os.path.join('./downloads/bert-wiki-ja_config', 'bert_finetuning_config_v1.json')
    checkpoint_file = os.path.join('./downloads/bert-wiki-ja', 'model.ckpt-1400000')

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
                    optimizer=Adam(),#'nadam',
                    metrics=['mae', 'mse', 'acc'])

    return model


train_dir = './datasets/finetuning/train'
test_dir = './datasets/finetuning/test'


# pandasでcsvの学習データとテストデータを読み込む
train_features_df = pd.read_csv(train_dir + '/features.csv')
train_labels_df = pd.read_csv(train_dir + '/labels.csv')
test_features_df = pd.read_csv(test_dir + '/features.csv')
test_labels_df = pd.read_csv(test_dir + '/labels.csv')

# 最大行の取得
maxlen_train = preprocessing.get_max(train_features_df['feature'])
maxlen_test = preprocessing.get_max(test_features_df['feature'])
print("maxlen_train", maxlen_train)
print("maxlen_test", maxlen_test)

maxlen=max(maxlen_train, maxlen_test)
print("maxlen", maxlen)

##### ラベル側の処理 #####

"""
label2index = {'positive': 0, 'negative': 1}
index2label = {0: 'positive', 1: 'negative'}
print("label2index", label2index)
print("index2label", index2label)
"""
#　クラス数（何種類に分類するか）ネガポジなら２
class_count = 2

train_dum = pd.get_dummies(train_labels_df)
test_dum = pd.get_dummies(test_labels_df)
train_labels = np.array(train_dum[['label_positive', 'label_negative']])
test_labels = np.array(test_dum[['label_positive', 'label_negative']])

print("train_labels　get_dummies :", train_labels.shape)
print("test_labels　get_dummies :", test_labels.shape)

##### 特徴量側の処理 #####

train_features = []
test_features = []
for feature in train_features_df['feature']:
    # ID化
    train_features.append(preprocessing.get_indice(feature, maxlen))
train_features = np.array(train_features)

# shape(len(train_features), maxlen)のゼロの行列作成
train_segments = np.zeros((len(train_features), maxlen), dtype = np.float32)

for feature in test_features_df['feature']:
    # ID化
    test_features.append(preprocessing.get_indice(feature, maxlen))
test_features = np.array(test_features)

# shape(len(test_features), maxlen)のゼロの行列作成
test_segments = np.zeros((len(test_features), maxlen), dtype = np.float32)

print("train_features :", train_features.shape)
print("test_features :", test_features.shape)


# データセットの作成
make_datasets.make_ds()

# パラメータ
SEQ_LEN = maxlen
BATCH_SIZE = 16
BERT_DIM = 768
LR = 1e-4

# 学習回数
EPOCH = 1#20

# 設定の変更
change_config.set_config(SEQ_LEN)

# モデルの作成
model = create_model()
model.summary()


# コールバック用　チェックポイント保存用
checkpoint_path = './models/finetuning_checkpoint'

# 学習
history = model.fit([train_features, train_segments],
          train_labels,
          epochs = EPOCH,#1,#3,
          #epochs = EPOCH,
          batch_size = BATCH_SIZE,
          validation_data=([test_features, test_segments], test_labels),
          shuffle=False,
          verbose = 1,
          callbacks = [
              ModelCheckpoint(monitor='val_acc', mode='max', filepath=checkpoint_path, save_best_only=True)
          ])

# モデルの保存
model.save('./models/saved_model_BERT')
