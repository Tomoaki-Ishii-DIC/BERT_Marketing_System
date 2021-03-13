import pandas as pd
import numpy as np
from preprocessing import preprocessing

#Predict

# テキストとテーブルデータを合わせたものを用意する
index_f_path = "./associated_data/dataframe_all_index.csv"
text_f_path = "./associated_data/dataframe_all_text.csv"
df_index = pd.read_csv(index_f_path)#, index_col=0
df_text = pd.read_csv(text_f_path)#, index_col=0


## df_s.loc[11]['text']
#
#X_f_path = "./associated_data/dataframe_all.csv"
#df_X_test = pd.read_csv(X_f_path, index_col=0)#, index_col=0, header=False
#print(df_X_test)
##df_X = df_X_test.tail(2)
##df_X = df_X_test[-2:][1:7]
#lstm_concat = pd.concat([df_X_test.iloc[:, 1:7], df_X_test.iloc[:, -1]], axis=1)

#print("df_X",df_X)
#print("df_X_test.shape", df_X_test.shape)
##print("lstm_concat", lstm_concat)
#print("lstm_concat.shape", lstm_concat.shape)
##print(df_X_test.columns.values)
#print("lstm_concat.columns.values", lstm_concat.columns.values)


inputs_list = []
label_list = []
#for df_values in df_X_test.columns.values:
for df_values in df_index.columns.values:
    if df_values == "label":
        continue
    else:
        # 入力が数値かどうかチェックする必要がある。
        input_data = int(input("Please Enter \"{}\": ".format(df_values)))
        #print("input_data", input_data)
        inputs_list.append(input_data)
    #if df_values == "1":
    #    break
    #elif df_values == "label":
    #    continue
    #elif df_values == "date":
    #    pass
    #    #input_data = input("Please Enter \"{}\"(YYYY-MM-DD): ".format(df_values))
    #    #inputs_list.append(input_data)
    #else:
    #    # 入力が数値かどうかチェックする必要がある。
    #    input_data = int(input("Please Enter \"{}\": ".format(df_values)))
    #    #print("input_data", input_data)
    #    inputs_list.append(input_data)

input_text = input("Please Enter \"Your press release text\": ")
#input_pred = pd.DataFrame([input_text])
#input_pred = []
#input_pred.append(input_text)
input_text = [input_text]
print(input_text)

print("inputs_list", len(inputs_list))
print(inputs_list)

#print("inputs_list", len(input_text))
#print(input_text)
# テキストをID化して足す
#pred_head = np.array(df.iloc[:10, 7:7+256])
#pred_tail = np.array(df.iloc[:10, -256:])
#pred_features = np.concatenate([train_head, train_tail], axis=1)


# 5000は変更できるようにする必要あり。
#maxを計算した上でIDにする
maxlen_text = preprocessing.get_max(input_text)
print("maxlen_text", maxlen_text)

if maxlen_text < 512:
    maxlen_text = 512

print("maxlen_text", maxlen_text)

pred_features = preprocessing.get_indice(input_text[0], maxlen_text)
#pred_features = preprocessing.get_indice(input_text, 512)

print("pred_features", pred_features)

#512対応
if maxlen_text > 512:
    pred_head = pred_features[:256]
    pred_tail = pred_features[-256:]
    pred_features = np.concatenate([pred_head, pred_tail])#, axis=1


print("pred_features", pred_features)


pred_features = pred_features[np.newaxis,:]
print(pred_features.shape)
print(pred_features)

maxlen = 512
pred_segments = np.zeros((len(pred_features), maxlen), dtype = np.float32)
#inputs_list.extend(text_id)

# ラベルを暫定で入力
inputs_list.append(0)

print(len(inputs_list))
print(inputs_list)

#train_features = []
#test_features = []
#for feature in train_features_df['feature']:
#    # 上で作った関数 _get_indice  を使ってID化
#    train_features.append(preprocessing._get_indice(feature, maxlen))
## shape(len(train_features), maxlen)のゼロの行列作成
#train_segments = np.zeros((len(train_features), maxlen), dtype = np.float32)

df_index_columns = df_index.columns.values
df_inputs = pd.DataFrame([inputs_list], columns=df_index_columns)#.T

#df_inputs.columns.values = df_X_test.columns.values
print("df_index", df_index[-2:].shape)
#print("lstm_concat", lstm_concat[-2:, 1:].shape)
print("df_inputs", df_inputs.shape)
#print(df_X)
#print(df_inputs)

X = pd.concat([df_index[-2:], df_inputs])
#X = pd.concat([lstm_concat[-2:, 1:], df_inputs[:, 1:]])
print(X)

# 日付とラベルの行を消す
#X = X.iloc[:, 1:-1]
# ラベルの行を消す
X = X.iloc[:, :-1]
print("X\n", X)
X_test = np.array(X)

print("X_test", X_test.shape)
print(X_test)

# LSTM用の形に直す
X_test = np.asarray(X_test).astype(np.float32)
X_test = X_test[np.newaxis,:,:]
print("X_test np.newaxis", X_test.shape)

#BERT(Predict)
#import types
#import sys
import pandas as pd
#import sentencepiece as spm
#import logging
import numpy as np
import os
import json
import openpyxl

#from keras import utils
from keras.models import load_model
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import get_custom_objects
#from sklearn.metrics import classification_report, confusion_matrix
from keras import Input, Model

from preprocessing import preprocessing
#from sklearn import preprocessing as sk_preprocessing


# モデルの読み込み・作成
model_path = './models/saved_model_BERT_part2'
model = load_model(model_path, custom_objects=get_custom_objects())

# model = Model(inputs=a, outputs=b) として"self_Attention"も出すようにする。
model = Model(inputs=model.input,
            outputs=[model.output,
            model.get_layer('Encoder-12-MultiHeadSelfAttention').output])

# ローカルJSONファイルの読み込み
json_path = "./downloads/bert-wiki-ja_config/bert_finetuning_config_v1.json"

# トークンの最大値を取得
with open(json_path) as f:
    data = json.load(f)
maxlen = data["max_position_embeddings"]
#print(maxlen)

predicted = model.predict([pred_features, pred_segments])
y_pred_BERT = predicted[0]
#y_pred_BERT = predicted[0].argmax(axis=1)

#y_pred_BERT = model.predict([pred_features, pred_segments])

print("y_pred_BERT", y_pred_BERT)

"""
# 推測処理とラベルの作成
y_train = []
for i in range(file_count):
    n_file = str(i+1).zfill(3)
    file_name = "features_" + n_file + ".csv"
    f_path = ("./datasets/pred_labeling/" + file_name)

    if not os.path.isfile(f_path):
        continue
    df_tests_features = pd.read_csv(f_path)

    print("File Name:{}の処理を開始しました。".format(file_name))

    # 出力用データフレーム作成
    df_sheet = pd.DataFrame()

    # 推測とAttentionの出力
    pred_list = []
    good_ratio_list = []
    for j in range(len(df_tests_features)):
        feature = df_tests_features.loc[j]['feature']

        test_features = []
        indices, tokens = preprocessing.get_indice_pred(feature, maxlen)
        test_features.append(indices)
        test_features = np.array(test_features)

        test_segments = np.zeros(
            (len(test_features), maxlen), dtype=np.float32)

        # model = Model(inputs=a, outputs=b)で推測とattentionを出力するようにしているため、
        # 出力は２次元のリスト predict[0][0]
        predicted = model.predict([test_features, test_segments])

        # 推測のみ変数へ保管
        y_pred = predicted[0].argmax(axis=1)

        # 平均を取るときに必要なため推測値０をマイナス１に変換する
        if y_pred[0] > 0.5:
            pred_list.append([1])
        else:
            pred_list.append([-1])


        # 高評価度算出（コメントにつけられたgoodとbadの比率）
        good = df_tests_features.loc[j]['good']
        bad = df_tests_features.loc[j]['bad']

        if bad == 0:
            # bad が ０ の時は計算できないので、代わりに2倍の値を入れる
            good_ratio = [good*2]
        else:
            good_ratio = [good/bad]

        good_ratio_list.append(good_ratio)

        # Attentionの抜き出し
        # model = Model(inputs=a, outputs=b) としてAttentionも出すようにしてあるため、
        # predicted[1]はAttention shape(1, maxlen ,BERT_DIM)
        # 平均だと値が小さくなるため最大値を取得
        weights = [w.max() for w in predicted[1][0]]

        # トークン(単語)とトークンに対応するAttention（最大値）からなるデータフレーム作成 shape(2, maxlen)
        df = pd.DataFrame([tokens, weights], index=['token', 'weight'])

        # Attentionの平均
        max_weight = np.asarray(weights).max()

        # タイプがNoneだとエラーが出るので０に置き換え
        df.loc['weight'] = df.loc['weight'].apply(lambda w: 0 if type(w) == type(None) else w)

        #行方向へ結合
        df_sheet = pd.concat([df_sheet, df])

    # インデックスのサイズを取得
    index_len = len(df_sheet.index)

    # pd.IndexSliceを使って指定した行に色をつけたいので、インデックスを数値に置き換える
    df_sheet = df_sheet.reset_index()
    # 元々のインデックスを削除
    df_sheet = df_sheet.drop('index', axis=1)

    # df.style.background_gradientで色つけ
    # pd.IndexSlice[1:index_len:2]で偶数行に対して処理を行う
    df_sheet = df_sheet.style.background_gradient(cmap='Reds', subset=pd.IndexSlice[1:index_len:2])

    # Excelに保管
    sheet_name = "comments_" + n_file
    with pd.ExcelWriter(excel_file, engine="openpyxl", mode="a") as writer:
        df_sheet.to_excel(writer, sheet_name=sheet_name, index=False)# , index=False, header=False

    # 高評価度を0から1の範囲に収まるように変更
    # サンプルが少ないときに0のせいでpredictの結果を極端に変えてしまうため無し。
    #good_ratio_list = sk_preprocessing.minmax_scale(good_ratio_list)

    # 高評価度ndarray
    good_ratio_list = np.array(good_ratio_list)

    # 推測値リスト（LSTMのラベルとして利用）
    pred_list = np.array(pred_list)
    y = pred_list*good_ratio_list

    if y.mean() > 0:
        y_train.append(1)
    else:
        y_train.append(0)

    print("File Name:{}の処理が完了しました。".format(file_name))

y_train_df = pd.DataFrame(y_train)
y_train_df.columns = ["label"]
y_train_df.to_csv("./datasets/y_train.csv")
print("y_train: ", y_train)

print("処理が完了しました。作成されたy_trainとattentionを確認してください。")
"""


#LSTM(Predict)

import pandas as pd
import numpy as np
from keras.models import load_model

from preprocessing import preprocessing


model = load_model('./models/saved_model_LSTM')
model.summary()

y_pred_LSTM = model.predict(X_test)
print("ネガポジ出力: ", y_pred_LSTM)

nega_posi = ['Positive', 'Negative']
y_pred = np.round(y_pred_LSTM).astype(int)[0,0]
print("ネガポジ予測: ", nega_posi[y_pred])


# アンサンブル
print("y_pred_BERT", y_pred_BERT)
print("y_pred_LSTM", y_pred_LSTM)

# 結果出力
y_pred = y_pred_BERT*0.5 + y_pred_LSTM*0.5
print("y_pred", y_pred)
y_pred_argmax = y_pred.argmax(axis=1)
print("y_pred", y_pred_argmax)
print("ネガポジ予測: ", nega_posi[y_pred_argmax[0]])
