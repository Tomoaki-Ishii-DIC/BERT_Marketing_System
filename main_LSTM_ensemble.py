import pandas as pd
import numpy as np
import os
import sys
import os
import pickle
import json

from keras_bert import load_trained_model_from_checkpoint

import keras
from keras.models import Model
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau#TensorBoard
from keras.optimizers import Adam

from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout

from sklearn.preprocessing import StandardScaler

from preprocessing import make_table
from preprocessing import preprocessing
from preprocessing import make_datasets
from preprocessing import change_config

from transformers import BertJapaneseTokenizer

def make_timestep_dataset(X_dataset, y_dataset, n_steps):
    """
    データセットをLSTMのタイムステップ入力に変形する関数

    Parameters
    ----------------
    X_dataset : ndarray, df  shape(samples, features)
        データセット（特徴量）
    y_dataset : ndarray, df  shape(samples,)
        ラベル
    n_steps : int
        タイムステップの数
    """
    X_list = []
    y_list = []

    time_steps = len(dataset) - n_steps + 1

    i = 0
    while i  < time_steps:
        end_idx = i + n_steps - 1
        X_seq = X_dataset[i:end_idx+1, :]
        y_seq = y_dataset[end_idx]

        X_list.append(X_seq)
        y_list.append(y_seq)

        i += 1

    X_ndarray = np.array(X_list)
    y_ndarray = np.array(y_list).astype(np.int32)

    return X_ndarray, y_ndarray


def create_model_BERT():
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
                    optimizer=Adam(),#'nadam',
                    metrics=['mae', 'mse', 'acc'])

    return model


# モデルの作成
def create_model_LSTM():
    """
    LSTMのモデルを作成する関数

    Parameters
    ----------------

    """
    model = Sequential()

    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(2, activation="softmax"))

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics = ['accuracy'])

    return model


# デーブルの作成（トレンドデータ、テキスト）
df_trend = make_table.trend()
df_news = make_table.text()

#　データセットの作成（トレンド＋指標データ、テキスト）
df_index, df_text = make_table.concat(df_trend, df_news)

# pred用に保存
df_index.to_csv('./datasets/df_index.csv', index=False)
df_text.to_csv('./datasets/df_text.csv', index=False)

##########　BERT(Train)　##########
print("")
print("BERT Model")

# validationデータの比率
VAL_SPLIT = 0.3

# trainデータのサンプル数の計算
n_sample = len(df_text)
val_samples = int(n_sample*VAL_SPLIT)
train_samples = int(n_sample-val_samples)

#サイズはBERTモデルの最大サイズ５１２に固定する
#maxlen=512

# トークンの最大値を取得
json_path = "./BERT-base_mecab-ipadic-bpe-32k/config.json"
with open(json_path) as f:
    data = json.load(f)
maxlen = data["max_position_embeddings"]

#テキストデータを入れるために使う空のテーブルを用意
ndarray_features = np.empty((len(df_text), maxlen))

# ラベルもカウントしてないか？
#一文ずつ抜き出して５１２を超えていたらHeadとTailだけ抜き出して結合する
for t in range(len(df_text)):
    token_len = len(df_text.iloc[t, :])
    zero_count = 0
    for col in reversed(range(len(df_text.iloc[t, :])-1)):
        if df_text.iloc[t, col] == 0.0:
            zero_count += 1
        else:
            break

    token_count = token_len - zero_count - 1
    if token_count > 512:
        ndarray_head = np.array(df_text.iloc[t, :256])
        ndarray_tail = np.array(df_text.iloc[t, token_count-256:token_count])
        df_resized = np.concatenate([ndarray_head, ndarray_tail])#, axis=1
    else:
        df_resized = np.array(df_text.iloc[t, :512])

    ndarray_features[t] = df_resized

# train_test_split
train_features = ndarray_features[:train_samples, :]
test_features = ndarray_features[train_samples:, :]

# labelをワンホット表現に変換
ndarray_labels = df_text.loc[:]['label']
labels_one_hot = np.identity(2)[ndarray_labels]
train_labels = labels_one_hot[:train_samples]
test_labels = labels_one_hot[train_samples:]

train_segments = np.zeros((len(train_features), maxlen), dtype = np.float32)
test_segments = np.zeros((len(test_features), maxlen), dtype = np.float32)


# パラメータ
SEQ_LEN = maxlen
BATCH_SIZE = 5
BERT_DIM = 768
LR = 1e-4
EPOCH = 20

# 設定の変更
#change_config.set_config(SEQ_LEN)

# モデルの作成
model_BERT = create_model_BERT()
model_BERT.summary()

# コールバック用
# チェックポイント保存先
checkpoint_path = './models/bert_checkpoint'
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
history = model_BERT.fit([train_features, train_segments],
                          train_labels,
                          epochs = EPOCH,
                          batch_size = BATCH_SIZE,
                          validation_data=([test_features, test_segments], test_labels),
                          shuffle=False,
                          verbose = 1,
                          callbacks = [check_point, early_stopping, reduce_lr])

# モデルの保存
model_BERT.save('./models/saved_model_BERT_part2')

##########　LSTM(Train)　##########

print("")
print("LSTM Model")

# タイムステップ数
TIMESTEPS = 5

dataset = np.array(df_index)

X_dataset = dataset[:, :-1]
y_dataset = dataset[:, -1]

#標準化
scaler = StandardScaler()
scaler.fit(X_dataset)
X_dataset_scaled = scaler.transform(X_dataset)

# テストで同じものを使用したいため保存
scalerfile = './StandardScaler.pkl'#.sav
pickle.dump(scaler, open(scalerfile, 'wb'))


# タイムステップを組み込んだLSTM用データセットの作成
X, y = make_timestep_dataset(X_dataset_scaled, y_dataset, TIMESTEPS)

# labelをワンホット表現に変換
y_one_hot = np.identity(2)[y]

# パラメータ
BATCHSIZE = 3
EPOCHS = 50
#VAL_SPLIT = 0.3#BERTと共通


# モデルの作成
model_LSTM = create_model_LSTM()

#コールバック用
early_stopping = EarlyStopping(monitor = "val_loss",
                                min_delta=0.001,
                                patience=7,
                                verbose=1,
                                mode="min",
                                restore_best_weights=False)

reduce_lr = ReduceLROnPlateau(monitor="val_loss",
                                factor=0.1,
                                patience=2,
                                verbose=1,
                                mode="min",
                                min_delta=0.0001)

history = model_LSTM.fit(X,
                        y_one_hot,
                        batch_size = BATCHSIZE,
                        epochs = EPOCHS,
                        validation_split=VAL_SPLIT,
                        callbacks=[early_stopping, reduce_lr])

model_LSTM.summary()

model_LSTM.save('./models/saved_model_LSTM')
