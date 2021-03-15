import pandas as pd
import numpy as np

# multivariate data preparation
#from numpy import array
#from numpy import hstack

#from numpy import array
#from numpy import hstack
#from numpy import insert,delete

import os

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers.recurrent import LSTM
from keras.layers import Dropout
from keras.preprocessing import sequence
from keras.optimizers import RMSprop

#from keras.preprocessing.sequence import TimeseriesGenerator
#from keras.models import Sequential
#from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
#import matplotlib.pyplot as plt


from preprocessing import make_table
#from preprocessing import make_table_text
#from preprocessing import make_table_trend
#from preprocessing import concat_dataset_table

# デーブルの作成（現行ではcsv出力。今後、dfの受け渡しに変更予定。）
df_text = make_table.text()
df_text
df_text["date"]

df_trend = make_table.trend()
df_trend

#concat_dataset_table.concat()
#make_table.concat()
df_table_index, df_table_text = make_table.concat(df_trend, df_text)

df_table_index
df_table_text

def make_timestep_dataset(dataset, n_steps):
    """
    データセットをLSTMのタイムステップ入力に変形する関数

    Parameters
    ----------------
    dataset : ndarray, df  shape(samples, features)
        データセット
    n_steps : int
        タイムステップの数
    """
    X_list = []
    y_list = []

    time_steps = len(dataset) - n_steps + 1

    i = 0
    while i  < time_steps:
        end_idx = i + n_steps - 1
        X_seq = dataset[i:end_idx+1, :-1]
        y_seq = dataset[end_idx, -1]

        X_list.append(X_seq)
        y_list.append(y_seq)

        i += 1

    X_ndarray = np.array(X_list)
    y_ndarray = np.array(y_list).astype(np.int32)

    return X_ndarray, y_ndarray


#新しい関数の出力を利用
df_index = df_table_index
df_text = df_table_text

#BERT(Train)
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
    #config_file = os.path.join('./downloads/bert-wiki-ja_config', 'bert_finetuning_config_v1.json')
    #checkpoint_file = os.path.join('./downloads/bert-wiki-ja', 'model.ckpt-1400000')
    config_file = os.path.join('./downloads/bert-wiki-ja_config', 'bert_lstm_config_v1.json')
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


print(df_text.iloc[:, :-1])
print(df_text.iloc[:, -1])

VAL_SPLIT = 0.3

n_sample = len(df_text)
print(n_sample)
val_samples = int(n_sample*VAL_SPLIT)
train_samples = int(n_sample-val_samples)

print(train_samples)
#print(val_samples)

#512対応
# 今は７とわかっているが、実際はその数値をどう取得するのか
#make_table.table()からかえす？
# １０についてもtrainとtestの比率を使う必要がある→train_samples
#train_features = np.array(df_text.iloc[:10, 7:7+512])
#test_features = np.array(df_text.iloc[10:, 7:7+512])

"""
train_head = np.array(df_text.iloc[:train_samples, :+256])
train_tail = np.array(df_text.iloc[:train_samples, -256:])
test_head = np.array(df_text.iloc[train_samples:, :+256])
test_tail = np.array(df_text.iloc[train_samples:, -256:])

train_features = np.concatenate([train_head, train_tail], axis=1)
test_features = np.concatenate([test_head, test_tail], axis=1)

print("train_features", train_features.shape)
print("test_features", test_features.shape)
"""
# ↑一文一文みないと
print("df_text", df_text.shape)

#print("df_text[t]", df_text.iloc[0, :])
"""
for t in range(len(df_text)):
    #print(df_text.iloc[t, :])
    num_count = df_text.iloc[t, :].value_counts()
    #print(num_count)#.sum()
    token_count = (df_text.iloc[t, :] > 0.0).sum()
    #print("zero_count", token_count)#.sum()
    if token_count > 512:
        print("head", df_text.iloc[t, :256])
        print("tail", df_text.iloc[t, token_count-256:token_count+2])
"""
#df_new = pd.DataFrame(index=df.index, columns=[])
#df_features = pd.DataFrame()
#df_features = pd.DataFrame(columns=range(1, 513))



# サイズはBERTモデルの最大サイズ５１２に固定する
maxlen=512

ndarray_features = np.empty((len(df_text), maxlen))
print("ndarray_features", ndarray_features.shape)
print("ndarray_features\n", ndarray_features)

#一文ずつ抜き出して５１２を超えていたらHeadとTailだけ抜き出して結合する
for t in range(len(df_text)):
    #for col in range(len(df_text.iloc[t, :])-1, , -1)range():
    token_len = len(df_text.iloc[t, :])
    print("token_len", token_len)
    zero_count = 0
    for col in reversed(range(len(df_text.iloc[t, :])-1)):
        #print("col", col)
        #print(df_text.iloc[t, :])
        if df_text.iloc[t, col] == 0.0:
            zero_count += 1
        else:
            break
    print("zero_count", zero_count)
    print("token_len-zero_count", token_len-zero_count)
        #print(num_count)#.sum()
        #token_count = (df_text.iloc[t, :] > 0.0).sum()
        #print("zero_count", token_count)#.sum()
    token_count = token_len - zero_count - 1
    if token_count > 512:
        ndarray_head = np.array(df_text.iloc[t, :256])
        ndarray_tail = np.array(df_text.iloc[t, token_count-256:token_count])
        print("ndarray_head", ndarray_head.shape)
        print("ndarray_tail",  ndarray_tail.shape)
        df_resized = np.concatenate([ndarray_head, ndarray_tail])#, axis=1
        #df_resized = pd.concat([df_head, df_tail], axis=1)#, axis=1
    else:
        df_resized = np.array(df_text.iloc[t, :512])

    print("df_resized",  df_resized.shape)
    print("ndarray_features",  ndarray_features.shape)
    #ndarray_features = np.concatenate([ndarray_features, df_resized])
    ndarray_features[t] = df_resized

print("ndarray_features", ndarray_features.shape)
print("ndarray_features\n", ndarray_features.astype(int))

# train_test_split
train_features = ndarray_features[:train_samples, :]
test_features = ndarray_features[train_samples:, :]

print("train_features", train_features.shape)
print("test_features", test_features.shape)



"""
#df_text.iloc[:train_samples, :+256]

train_head = np.array(df_text.iloc[:train_samples, :+256])
train_tail = np.array(df_text.iloc[:train_samples, -256:])
test_head = np.array(df_text.iloc[train_samples:, :+256])
test_tail = np.array(df_text.iloc[train_samples:, -256:])

print("train_head", train_head.shape)
print("train_tail", train_tail.shape)
print("test_head", test_head.shape)
print("test_tail", test_tail.shape)

train_features = np.concatenate([train_head, train_tail], axis=1)
test_features = np.concatenate([test_head, test_tail], axis=1)

print("train_features", train_features.shape)
print("test_features", test_features.shape)
"""


# labelをワンホット表現に変形
ndarray_labels = df_text.iloc[:, -1]
labels_one_hot = np.identity(2)[ndarray_labels]
print("labels_one_hot", labels_one_hot)
train_labels = labels_one_hot[:train_samples]
test_labels = labels_one_hot[train_samples:]
print("train_features", train_features)
print("train_labels", train_labels)
print("test_features", test_features)
print("test_labels", test_labels)


train_segments = np.zeros((len(train_features), maxlen), dtype = np.float32)
test_segments = np.zeros((len(test_features), maxlen), dtype = np.float32)

print(train_features.shape)
print(train_segments.shape)
print(train_labels.shape)
print(test_features.shape)
print(test_segments.shape)
print(test_labels.shape)


# データセットの作成
#make_datasets.make_ds()

# パラメータ
SEQ_LEN = maxlen
BATCH_SIZE = 5
BERT_DIM = 768
LR = 1e-4

# 学習回数
EPOCH = 3#20

# 設定の変更
change_config.set_config(SEQ_LEN)

# モデルの作成
model = create_model()
model.summary()

train_features.shape

# コールバック用　チェックポイント保存用
checkpoint_path = './models/finetuning_checkpoint_2'


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
model.save('./models/saved_model_BERT_part2')


#LSTM(Train)

# choose a number of time steps
n_steps = 3
#split_rate = 0.3

#dataset = np.array(df.iloc[:, 1:])
dataset = np.array(df_index)
print("dataset.shape", dataset.shape)

# convert into input/output

X, y = make_timestep_dataset(dataset, n_steps)
##X, y = make_timestep_dataset(dataset, n_steps)

print(X.shape, y.shape)
#print(X)
print(y)
#print(type(y[0]))


y_one_hot = np.identity(2)[y]
y_one_hot

BATCHSIZE = 3
EPOCHS = 50
#VAL_SPLIT = 0.3#上に移動

model = Sequential()

model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(2, activation="softmax"))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics = ['accuracy'])

split_rate = 0.7
split = int(X.shape[0]*split_rate)

print("Split rate: ", split_rate)
print("Train_split: ", split)
print("Val_split: ", X.shape[0]-split)

#X_train = X[:split, :]
#X_test = X[split:, :]
#y_train = y[:split]
#y_test = y[split:]
#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)


early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=0)
history = model.fit(X, y_one_hot, batch_size = BATCHSIZE, epochs = EPOCHS, validation_split=0.3, callbacks=[early_stopping])
#history = model.fit(X_train, y_train, batch_size = BATCHSIZE, epochs = EPOCHS)
#modelName = model.__class__.__name__


model.save('./models/saved_model_LSTM')
