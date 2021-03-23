import pandas as pd
import numpy as np
import os
import json
import openpyxl
import pickle

from keras_bert import load_trained_model_from_checkpoint
from keras_bert import get_custom_objects

from keras import Input, Model
from keras.models import load_model

from preprocessing import preprocessing
from preprocessing import make_table


## デーブルの作成（トレンドデータ、テキスト）
#df_trend = make_table.trend()
#df_news = make_table.text()

#　データセットの作成（トレンド＋指標データ、テキスト）
#df_index, df_text = make_table.concat(df_trend, df_news)

# 速度向上のため、csvから読み込む
df_index = pd.read_csv('./datasets/df_index.csv')
df_text = pd.read_csv('./datasets/df_text.csv')



import PySimpleGUI as sg
import time
#################### PySimpleGUI用コード ####################
columns_list = df_index.columns.values[:-1]

#  セクション1 - オプションの設定と標準レイアウト
sg.theme('Dark Blue 3')

# sg.Input(key='-INPUT-')
inputs_list = []
inputs_list.append([sg.Text('Python GUI')])

for columns_name in columns_list:
    inputs_list.append([sg.Text(columns_name, size=(15, 1)), sg.Input(key=columns_name)])

inputs_list.append([sg.Text('テキスト', size=(15, 7)), sg.Multiline(size=(45, 7), key='テキスト')])

inputs_list.append([sg.Submit(button_text='実行ボタン')])
inputs_list.append([sg.Text(key='-OUTPUT-', size=(50,1))])#

# セクション 2 - ウィンドウの生成
window = sg.Window('入力', inputs_list)

# セクション 3 - イベントループ
while True:
    event, values = window.read()

    if event is None:
        print('exit')
        break

    if event == '実行ボタン':


        #for input in inputs_list:
        #    print("inputs_list", input)
        print("values['テキスト']", values['テキスト'])
        show_message = ""

        #不要。代わりにpred結果を入れる
        #for columns_name in columns_list:
        #    show_message += columns_name + "：" + values[columns_name] + 'が入力されました。\n'
        #
        #show_message += "テキスト：" + values['テキスト'] + 'が入力されました。\n'


        # Output a message to the window
        window['-OUTPUT-'].update("上記内容で処理中です。完了までしばらくお待ちください。", text_color='yellow')

        #################### predict処理 ####################
        input_text = values['テキスト']
        input_text = [input_text]
        print("####input_text####", input_text)
        maxlen_text = preprocessing.get_max(input_text)

        if maxlen_text < 512:
            maxlen_text = 512

        pred_features = preprocessing.get_indice(input_text[0], maxlen_text)

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
        print("df_index_columns", df_index_columns)
        print("values", inputs_values)

        #inputs_list_2 = []
        #for input in columns_list:
        #    inputs_list_2.append(inputs_list[input])
        #print("inputs_list", inputs_list_2)
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

        # ローカルJSONファイルの読み込み
        json_path = "./downloads/bert-wiki-ja_config/bert_finetuning_config_v1.json"

        # トークンの最大値を取得
        with open(json_path) as f:
            data = json.load(f)
        maxlen = data["max_position_embeddings"]

        predicted = model_BERT.predict([pred_features, pred_segments])
        y_pred_BERT = predicted[0]


        #LSTM(Predict)
        model_LSTM = load_model('./models/saved_model_LSTM')

        y_pred_LSTM = model_LSTM.predict(X_test)


        # アンサンブル
        print()
        print("BERT 予測確率:Positive {:f}, Negative {:f}".format(y_pred_BERT[0][0], y_pred_BERT[0][1]))
        print("LSTM 予測確率:Positive {:f}, Negative {:f}".format(y_pred_LSTM[0][0], y_pred_LSTM[0][1]))
        pred_message = ""
        pred_message += "BERT 予測確率: Positive {:f}, Negative {:f}\n".format(y_pred_BERT[0][0], y_pred_BERT[0][1])
        pred_message += "LSTM 予測確率: Positive {:f}, Negative {:f}\n".format(y_pred_LSTM[0][0], y_pred_LSTM[0][1])
        # 結果出力
        y_pred = y_pred_BERT*0.5 + y_pred_LSTM*0.5
        print("Ensemble 確率:Positive {:f}, Negative {:f}".format(y_pred[0][0], y_pred[0][1]))
        pred_message += "Ensemble 確率: Positive {:f}, Negative {:f}\n".format(y_pred[0][0], y_pred[0][1])
        y_pred_argmax = y_pred.argmax(axis=1)
        nega_posi = ['Positive', 'Negative']
        print()
        print("ネガポジ予測:", nega_posi[y_pred_argmax[0]])
        pred_message += "\nネガポジ予測: {}\n".format(nega_posi[y_pred_argmax[0]])

        #################### PySimpleGUI用コード ####################
        window['-OUTPUT-'].update("")

        # ポップアップ　結果
        sg.popup(pred_message)


    #一番下に入れないと画面が閉じなくなる
    #if event == sg.WINDOW_CLOSED or event == 'Quit':
    #    break


# セクション 4 - ウィンドウの破棄と終了
window.close()


"""
inputs_list = []
for df_values in df_index.columns.values:
    if df_values == "label":
        continue
    else:
        # 入力が数値かどうかチェックする必要がある。
        input_data = int(input("Please Enter \"{}\": ".format(df_values)))
        inputs_list.append(input_data)

input_text = input("Please Enter \"Your press release text\": ")
input_text = [input_text]
print("####input_text####", input_text)
#input_text = values['テキスト']

maxlen_text = preprocessing.get_max(input_text)

if maxlen_text < 512:
    maxlen_text = 512

pred_features = preprocessing.get_indice(input_text[0], maxlen_text)


#512対応
if maxlen_text > 512:
    pred_head = pred_features[:256]
    pred_tail = pred_features[-256:]
    pred_features = np.concatenate([pred_head, pred_tail])#, axis=1

pred_features = pred_features[np.newaxis,:]

maxlen = 512
pred_segments = np.zeros((len(pred_features), maxlen), dtype = np.float32)

# ラベルを暫定で入力
inputs_list.append(0)

df_index_columns = df_index.columns.values
df_inputs = pd.DataFrame([inputs_list], columns=df_index_columns)#.T

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

# ローカルJSONファイルの読み込み
json_path = "./downloads/bert-wiki-ja_config/bert_finetuning_config_v1.json"

# トークンの最大値を取得
with open(json_path) as f:
    data = json.load(f)
maxlen = data["max_position_embeddings"]

predicted = model_BERT.predict([pred_features, pred_segments])
y_pred_BERT = predicted[0]


#LSTM(Predict)
model_LSTM = load_model('./models/saved_model_LSTM')

y_pred_LSTM = model_LSTM.predict(X_test)


# アンサンブル
print()
print("BERT 予測確率:Positive {:f}, Negative {:f}".format(y_pred_BERT[0][0], y_pred_BERT[0][1]))
print("LSTM 予測確率:Positive {:f}, Negative {:f}".format(y_pred_LSTM[0][0], y_pred_LSTM[0][1]))

# 結果出力
y_pred = y_pred_BERT*0.5 + y_pred_LSTM*0.5
print("Ensemble 確率:Positive {:f}, Negative {:f}".format(y_pred[0][0], y_pred[0][1]))

y_pred_argmax = y_pred.argmax(axis=1)
nega_posi = ['Positive', 'Negative']
print()
print("ネガポジ予測:", nega_posi[y_pred_argmax[0]])
"""
