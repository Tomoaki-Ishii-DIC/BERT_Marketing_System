import pandas as pd
import numpy as np
import os
import json
import openpyxl
import pickle

import PySimpleGUI as sg

from keras_bert import load_trained_model_from_checkpoint
from keras_bert import get_custom_objects

from keras import Input, Model
from keras.models import load_model

from preprocessing import preprocessing
from preprocessing import make_table

from transformers import BertJapaneseTokenizer

############ デーブルの作成（トレンドデータ、テキスト）############
#df_trend = make_table.trend()
#df_news = make_table.text()

#　データセットの作成（トレンド＋指標データ、テキスト）
#df_index, df_text = make_table.concat(df_trend, df_news)

# 速度向上のため、csvから読み込む
df_index = pd.read_csv('./datasets/df_index.csv')
df_text = pd.read_csv('./datasets/df_text.csv')

#################### ↓PySimpleGUI用コード ####################
columns_list = df_index.columns.values[:-1]

inputs_list = []
inputs_list.append([sg.Text('【 変数名 】', size=(20, 1)), sg.Text('【 入力 】', size=(20, 1))])

for columns_name in columns_list:
    inputs_list.append([sg.Text(columns_name, size=(20, 1)), sg.Input(key=columns_name)])

inputs_list.append([sg.Text('テキスト', size=(20, 7)), sg.Multiline(size=(45, 7), key='テキスト')])
inputs_list.append([sg.Button('実行'), sg.Button('終了')])
#inputs_list.append([sg.Text(key='-OUTPUT-', size=(50,1))])

window = sg.Window('推測用データ入力', inputs_list)

tknz = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')

while True:
    event, values = window.read()

    if event is None:
        print('exit')
        break
    elif event == sg.WINDOW_CLOSED or event == '終了':
        break
    elif event == '実行':
        #################### ↓predict処理 ####################
        input_text = values['テキスト']
        input_text = [input_text]

        maxlen_text = preprocessing.get_max(input_text)

        if maxlen_text < 512:
            maxlen_text = 512

        pred_features = preprocessing.get_indice(input_text[0], maxlen_text, tknz)

        #512対応
        if maxlen_text > 512:
            pred_head = pred_features[:256]
            pred_tail = pred_features[-256:]
            pred_features = np.concatenate([pred_head, pred_tail])#, axis=1

        pred_features = pred_features[np.newaxis,:]

        maxlen = 512
        pred_segments = np.zeros((len(pred_features), maxlen), dtype = np.float32)


        # 入力をlistに変換
        inputs_values = list(values.values())
        inputs_values.pop(-1)

        # ラベルを暫定で入力
        inputs_values.append(0)

        #df_index_columns = df_index.columns.values[:-1]
        df_index_columns = df_index.columns.values

        df_inputs = pd.DataFrame([inputs_values], columns=df_index_columns)#.T

        # trainと合わせる
        TIMESTEPS = 5
        X = pd.concat([df_index[-(TIMESTEPS-1):], df_inputs])

        # ラベルの行を消す
        X = X.iloc[:, :-1]
        X_test = np.array(X)

        #標準化（学習時のインスタンスを使用する）
        scalerfile = './StandardScaler.pkl'#.sav
        scaler = pickle.load(open(scalerfile, 'rb'))

        scaler.fit(X_test)
        X_test_scaled = scaler.transform(X_test)

        # LSTM用の形に直す
        X_test = np.asarray(X_test_scaled).astype(np.float32)
        X_test = X_test[np.newaxis,:,:]


        #BERT(Predict)
        # モデルの読み込み・作成
        model_path = './models/saved_model_BERT_part2'
        model_BERT = load_model(model_path, custom_objects=get_custom_objects())

        # model = Model(inputs=a, outputs=b) として"self_Attention"も出すようにする。
        model_BERT = Model(inputs=model_BERT.input,
                            outputs=[model_BERT.output,
                            model_BERT.get_layer('Encoder-12-MultiHeadSelfAttention').output])

        # トークンの最大値を取得
        json_path = './BERT-base_mecab-ipadic-bpe-32k/config.json'
        with open(json_path) as f:
            data = json.load(f)
        maxlen = data["max_position_embeddings"]

        predicted = model_BERT.predict([pred_features, pred_segments])
        y_pred_BERT = predicted[0]


        #LSTM(Predict)
        model_LSTM = load_model('./models/saved_model_LSTM')
        y_pred_LSTM = model_LSTM.predict(X_test)

        # 出力テキスト用変数
        pred_message = ""

        # 結果出力
        print()
        temp_message = "BERT 予測確率: Positive {:f}, Negative {:f}".format(y_pred_BERT[0][0], y_pred_BERT[0][1])
        print(temp_message)
        pred_message += temp_message + "\n"

        temp_message = "LSTM 予測確率: Positive {:f}, Negative {:f}".format(y_pred_LSTM[0][0], y_pred_LSTM[0][1])
        print(temp_message)
        pred_message += temp_message + "\n"

        # アンサンブル
        y_pred = y_pred_BERT*0.5 + y_pred_LSTM*0.5

        temp_message = "Ensemble 確率: Positive {:f}, Negative {:f}".format(y_pred[0][0], y_pred[0][1])
        print(temp_message)
        pred_message += temp_message + "\n"

        nega_posi = ['Positive', 'Negative']
        y_pred_argmax = y_pred.argmax(axis=1)
        temp_message = "ネガポジ予測: {}".format(nega_posi[y_pred_argmax[0]])
        print()
        print(temp_message)
        pred_message += "\n" + temp_message + "\n"

        #################### ↓PySimpleGUI用コード ####################
        # 結果表示
        sg.popup(pred_message, title = '出力結果')

window.close()
