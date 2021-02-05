import types

import sys
import pandas as pd
import sentencepiece as spm
import logging
import numpy as np
import os

from keras import utils
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import get_custom_objects
from sklearn.metrics import classification_report, confusion_matrix
from keras import Input, Model

import openpyxl

import preprocessing

#sys.pathに追加（必要なのか調査が必要）
sys.path.append('modules')

# 上にあったのと同じ？ → predict用に変更
def _get_indice_pred(feature, maxlen):
    indices = np.zeros((maxlen), dtype=np.int32)

    tokens = []
    tokens.append('[CLS]')
    pre_text = preprocessing.preprocessing_text(feature)#追加
    tokenized_text = preprocessing.tokenizer_mecab(pre_text)#追加
    tokens.extend(tokenized_text)#追加
    #tokens.extend(spp.encode_as_pieces(feature))
    tokens.append('[SEP]')

    for t, token in enumerate(tokens):
        if t >= maxlen:
            break
        try:
            indices[t] = spp.piece_to_id(token)
        except:
            logging.warn('unknown')
            indices[t] = spp.piece_to_id('<unk>')

    return indices, tokens

#tests_features_df = pd.read_csv('./datasets/pred_labeling/features_006.csv')
#tests_features_df.loc[0]['feature']

# SentencePieceProccerモデルの読込
spp = spm.SentencePieceProcessor()
spp.Load('./downloads/bert-wiki-ja/wiki-ja.model')

# BERTの学習したモデルの読込（ダウンロードした？勝手に保存される？）
model_filename = './downloads/models/knbc_finetuning.model'
model = load_model(model_filename, custom_objects=get_custom_objects())
#model = load_model(model_filename, custom_objects=SeqSelfAttention.get_custom_objects())
model = Model(inputs=model.input, outputs=[model.output, model.get_layer('Encoder-12-MultiHeadSelfAttention').output])
# ↑ここでmodel = Model(inputs=a, outputs=b) としてAttentionも出すようにする。


# 上のと同じのを入れると思われるため、消していいかも(ファイルを分けるなら必要)
#SEQ_LEN = 103#206
#import preprocessing#自作ファイルの読み込み
#max_numbers = preprocessing.get_max(train_features_df['feature'])
SEQ_LEN = 224#max_token_num#max_numbers
maxlen = SEQ_LEN

file_count = sum((len(f) for _, _, f in os.walk("./datasets/pred_labeling"))) - 1
print(file_count)
y_train = []
for i in range(file_count):
#for i in range(3):
    n_file = str(i+1).zfill(3)
    file_name = "features_" + n_file + ".csv"
    f_path = ("./datasets/pred_labeling/" + file_name)
    if not os.path.isfile(f_path):
        continue

    df_tests_features = pd.read_csv(f_path)
    print("File Name: ", file_name)

    #excelファイル保管用
    excel_file = './attention_excel/attention_' + n_file + '.xlsx'
    writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')

    wb = openpyxl.Workbook()
    sheet = wb.active
    sheet.title = 'Cover'
    c1 = sheet["A1"]
    c2 = sheet["A2"]
    c1.value =  "Attention出力用ファイルです。"
    c2.value =  "詳細は次のシート以降を参照してください。"
    wb.save(excel_file)


    pred_list = []
    good_ratio_list = []
    for j in range(len(df_tests_features)):
#    for j in range(2):
        feature = df_tests_features.loc[j]['feature']

        test_features = []
        indices, tokens = _get_indice_pred(feature, maxlen)
        test_features.append(indices)

        #勝手に追加
        test_features = np.array(test_features)

        test_segments = np.zeros(
            (len(test_features), maxlen), dtype=np.float32)

        # model = Modelを使えば推定　predict[0][0]　２次元のリストで返せる。
        predicted = model.predict([test_features, test_segments])#.argmax(axis=1)
        #predict = model.predict(test_features)

        y_pred = predicted[0].argmax(axis=1)
        #print("tokens: ", tokens)
        #print("predict: ", y_pred[0])

        if y_pred[0] > 0.5:
            pred_list.append([1])
        else:
            pred_list.append([-1])


        # 高評価度算出
        good = df_tests_features.loc[j]['good']
        bad = df_tests_features.loc[j]['bad']

        if bad == 0:
            good_ratio = [0]
        else:
            good_ratio = [good/bad]

        good_ratio_list.append(good_ratio)


        # 入力シーケンスはpad_sequenceにより、以下の様に0でpre paddingしています。
        # [0 0 0 0 x1(300) x2(300) x3(300)] ←３００は (None, 11, 300) の
        # Attention Weightは入力シーケンスに対応して計算されるため、
        # 入力シーケンスのpadding分シフトします。
        #weights = [w.max() for w in predicted[1][0][-len(tokens):]]
        weights = [w.max() for w in predicted[1][0]]#[-len(tokens):]
        df = pd.DataFrame([tokens, weights], index=['token', 'weight']).T

        mean = np.asarray(weights).mean()#np.asarray　参照コピー

        df['rank'] = df['weight'].rank(ascending=False)#ランキング
        # wから平均を引いた値が0より大きいものだけ（偏差）
        df['normalized'] = df['weight'].apply(lambda w: 0 if type(w) == type(None) else max(w - mean, 0))
        #df['normalized'] = df['weight'].apply(lambda w: max(w - mean, 0))#行全体や列全体に対して、同じ操作
        df['weight'] = df['weight'].astype('float32')
        df['attention'] = df['normalized'] > 0
        # df.style.background_gradient で色つけ
        df = df.style.background_gradient(cmap='Blues', subset=['normalized'])

        # excel に保存
        sheetname="comment" + str(j)
        with pd.ExcelWriter(excel_file, engine="openpyxl", mode="a") as writer:
            df.to_excel(writer, sheet_name=sheetname, index=False)

        #display(df)

    # 加重平均が０より大きいか
    y = np.array(pred_list)*np.array(good_ratio_list)

    if y.mean() > 0:
        y_train.append(1)
    else:
        y_train.append(0)

    print("pred_list: ", pred_list)
    print("good_ratio_list: ", good_ratio_list)
    print()

y_train_df = pd.DataFrame(y_train)
y_train_df.columns = ["label"]
y_train_df.to_csv("./datasets/y_train.csv")
print("y_train: ", y_train)
