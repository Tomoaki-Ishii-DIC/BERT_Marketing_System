import sys
from keras_bert import load_trained_model_from_checkpoint
import pandas as pd
import sentencepiece as spm

import preprocessing

#sys.pathに追加（必要なのか調査が必要）
sys.path.append('modules')

#import pprint
#pprint.pprint(sys.path)





from keras.utils import np_utils
from keras import utils
import numpy as np

# ここでもsentence piece

# 上に同じ名前の関数があるので注意
# 最大単語数分のID化された文を返す関数
# maxlenがなくてエラーになるので勝手に追加（maxlenは最大単語数か？）


# 関数移動　_get_indice


#勝手に追加 maxlen=103
# 引数のパスは直接書けばいらないかも
def _load_labeldata(train_dir, test_dir, maxlen):
    # pandasでcsvの学習データとテストデータを読み込む
    train_features_df = pd.read_csv(f'{train_dir}/features.csv')
    train_labels_df = pd.read_csv(f'{train_dir}/labels.csv')
    test_features_df = pd.read_csv(f'{test_dir}/features.csv')
    test_labels_df = pd.read_csv(f'{test_dir}/labels.csv')

    ##### ラベル側の処理 #####

    # ラベルのユニーク値を取り出す（ラベル数）（インデックスとラベル別別に保管）
    # ネガポジなら　ポジティブ, ネガティブ と　０、１　を入れてしまえばいいと思われる
    #{'スポーツ': 0, '携帯電話': 1},
    label2index = {k: i for i, k in enumerate(train_labels_df['label'].unique())}
    #{0: 'スポーツ', 1: '携帯電話'}
    index2label = {i: k for i, k in enumerate(train_labels_df['label'].unique())}
    #　クラス数（何種類に分類するか）ネガポジなら２
    class_count = len(label2index)

    # Numpyユーティリティ to_categorical(y, nb_classes=None)
    # クラスベクトル（0からnb_classesまでの整数）を categorical_crossentropyとともに用いるためのバイナリのクラス行列に変換します．
    # y: 行列に変換するクラスベクトル, nb_classes: 総クラス数
    # ↓trainのラベルを文字からインデックスを使用して変換
    train_labels = utils.np_utils.to_categorical([label2index[label] for label in train_labels_df['label']], num_classes=class_count)
    #　testのインデックスをまず作る
    test_label_indices = [label2index[label] for label in test_labels_df['label']]
    # ↓testのラベルを文字からインデックスを使用して変換
    test_labels = utils.np_utils.to_categorical(test_label_indices, num_classes=class_count)

    ##### 特徴量側の処理 #####

    train_features = []
    test_features = []
    for feature in train_features_df['feature']:
        # 上で作った関数 _get_indice  を使ってID化
        train_features.append(preprocessing._get_indice(feature, maxlen))
    # shape(len(train_features), maxlen)のゼロの行列作成
    train_segments = np.zeros((len(train_features), maxlen), dtype = np.float32)

    for feature in test_features_df['feature']:
        # 上で作った関数 _get_indice  を使ってID化
        test_features.append(preprocessing._get_indice(feature, maxlen))
    # shape(len(test_features), maxlen)のゼロの行列作成
    test_segments = np.zeros((len(test_features), maxlen), dtype = np.float32)

    print(f'Trainデータ数: {len(train_features_df)}, Testデータ数: {len(test_features_df)}, ラベル数: {class_count}')

    return {
        'class_count': class_count,
        'label2index': label2index,
        'index2label': index2label,
        'train_labels': train_labels,
        'test_labels': test_labels,
        'test_label_indices': test_label_indices,
        'train_features': np.array(train_features),
        'train_segments': np.array(train_segments),
        'test_features': np.array(test_features),
        'test_segments': np.array(test_segments),
        'input_len': maxlen
    }



# おそらく、この関数を作った理由は複数分類モデルを自由に作れるようにしたかったからだ。
# 単にネガポジにするなら関数にしないで直接書けばいい。

from keras.layers import Dense, Dropout, LSTM, Bidirectional, Flatten, GlobalMaxPooling1D
from keras_bert.layers import MaskedGlobalMaxPool1D
from keras import Input, Model

# nadam を選べば使わなくてもいい
# https://github.com/CyberZHG/keras-bert
from keras_bert import AdamWarmup, calc_train_steps

def _create_model(input_shape, class_count):
    # AdamWarmupをオプティマイザーとして使用するために必要な情報を得る関数
    # nadam を選べば使わなくてもいい
    decay_steps, warmup_steps = calc_train_steps(
        input_shape[0],
        batch_size=BATCH_SIZE,
        epochs=EPOCH,
    )

    # 学習済みモデル 「bert」 の最終出力層のoutputを取り出す
    bert_last = bert.get_layer(name='NSP-Dense').output
    x1 = bert_last
    # 最終出力層のoutputを新規作成した全結合層に入れる
    output_tensor = Dense(class_count, activation='softmax')(x1)

    # Trainableの場合は、Input Masked Layerが3番目の入力なりますが、
    # FineTuning時には必要無いので1, 2番目の入力だけ使用します。
    # Trainableでなければkeras-bertのModel.inputそのままで問題ありません。
    model = Model([bert.input[0], bert.input[1]], output_tensor)
    model.compile(loss='categorical_crossentropy',
                  optimizer=AdamWarmup(decay_steps=decay_steps, warmup_steps=warmup_steps, lr=LR),
                  #optimizer='nadam',
                  metrics=['mae', 'mse', 'acc'])

    return model

# データロードとモデルの準備
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard




# BERTのロード
config_path = './downloads/bert-wiki-ja_config/bert_finetuning_config_v1.json'
# 拡張子まで記載しない（.ckptファイルで保存されている）
checkpoint_path = './downloads//bert-wiki-ja/model.ckpt-1400000'

# 最大のトークン数
#import preprocessing#自作ファイルの読み込み
#max_numbers = preprocessing.get_max(train_features_df['feature'])
SEQ_LEN = 224#max_token_num#max_numbers
BATCH_SIZE = 16
BERT_DIM = 768
LR = 1e-4
# 学習回数
EPOCH = 1#20

# 学習ずみモデルでモデル構築
bert = load_trained_model_from_checkpoint(config_path, checkpoint_path, training=True,  trainable=True, seq_len=SEQ_LEN)
bert.summary()

# この後に追加する（転移学習）
# 分類問題用にモデルの再構築


trains_dir = './datasets/finetuning/train'
tests_dir = './datasets/finetuning/test'

#上で作った関数
data = _load_labeldata(trains_dir, tests_dir, SEQ_LEN)

# モデルの読み込み
model_filename = './downloads/models/knbc_finetuning.model'

# 上で作った関数（関数を使わずに直接書くこともできる）
# data['train_features'].shape　は　文の数×最大単語数　＝　特徴量Xのインプットshape
# data['class_count']　は　クラスの数
model = _create_model(data['train_features'].shape, data['class_count'])

model.summary()



history = model.fit([data['train_features'], data['train_segments']],
          data['train_labels'],
          epochs = EPOCH,#1,#3,
          #epochs = EPOCH,
          batch_size = BATCH_SIZE,
          validation_data=([data['test_features'], data['test_segments']], data['test_labels']),
          shuffle=False,
          verbose = 1,
          callbacks = [
              ModelCheckpoint(monitor='val_acc', mode='max', filepath=model_filename, save_best_only=True)
          ])
