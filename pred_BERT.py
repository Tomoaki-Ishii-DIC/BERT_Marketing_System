import pandas as pd
import numpy as np
import os
import json
import openpyxl

from keras.models import load_model
from keras_bert import load_trained_model_from_checkpoint
from keras_bert import get_custom_objects
from keras import Input, Model

from preprocessing import preprocessing


# モデルの読み込み・作成
model_path = './models/saved_model_BERT'
model = load_model(model_path, custom_objects=get_custom_objects())

# outputs=[model.output,model.get_layer]とすることで"self_Attention"も出力できるようにする
model = Model(inputs=model.input,
              outputs=[model.output,
                        model.get_layer('Encoder-12-MultiHeadSelfAttention').output])

# ローカルJSONファイルの読み込み
json_path = "./downloads/bert-wiki-ja_config/bert_finetuning_config_v1.json"

# トークンの最大値を取得
with open(json_path) as f:
    data = json.load(f)
maxlen = data["max_position_embeddings"]

# csvファイルの数を取得
csv_path_fine_train = ("./datasets_csv/finetuning/train")
csv_path_fine_test = ("./datasets_csv/finetuning/test")
csv_path_pred_labeling = ("./datasets_csv/pred_labeling")

csv_folder = [csv_path_fine_train , csv_path_fine_test, csv_path_pred_labeling]

file_count = 0
for p in csv_folder:
    dir = p + '/comments'
    file_count += len([name for name in os.listdir(dir) if name[-4:] == '.csv'])


# excelファイルの準備
excel_file = './attention_excel/self_attention.xlsx'
writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')
wb = openpyxl.Workbook()
sheet = wb.active

# 表紙の作成
sheet.title = 'Cover'
c1 = sheet["A1"]
c2 = sheet["A2"]
c1.value =  "Attention出力用ファイルです。"
c2.value =  "詳細は次のシート以降を参照してください。"

# excelファイルの保存
wb.save(excel_file)

# 推測処理とラベルの作成
y_train = []
for i in range(file_count):
    n_file = str(i+1).zfill(3)
    #file_name = "features_" + n_file + ".csv"
    file_name = "comment_dataset_" + n_file + ".csv"
    f_path = ("./datasets_csv/pred_labeling/comments/" + file_name)

    if not os.path.isfile(f_path):
        continue
    df_tests_features = pd.read_csv(f_path, index_col=0)

    print("File Name:{}の処理を開始しました。".format(file_name))

    #print("入力確認 df_tests_features", df_tests_features)
    print("入力確認 df_tests_features.shape", df_tests_features.shape)

    # 出力用データフレーム作成
    df_sheet = pd.DataFrame()

    # 推測とAttentionの出力
    pred_list = []
    good_ratio_list = []
    for j in range(len(df_tests_features)):
        #feature = df_tests_features.loc[j]['feature']
        feature = df_tests_features.loc[j]['text']

        test_features = []
        indices, tokens = preprocessing.get_indice_pred(feature, maxlen)
        test_features.append(indices)
        test_features = np.array(test_features)

        test_segments = np.zeros(
            (len(test_features), maxlen), dtype=np.float32)

        # 推測値とattentionを出力するようにしているため、
        # 出力は２次元のリスト predict[a][b]
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
        # predictでAttentionも出すようにしてあり、predicted[1]はAttention shape(1, maxlen ,BERT_DIM)
        # 平均してしまうと値が小さくなりすぎるため最大値を取得
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

    #print("pred_list: ", pred_list)
    print("pred_list.shape: ", pred_list.shape)
    #print("good_ratio_list: ", good_ratio_list)

y_train_df = pd.DataFrame(y_train)
y_train_df.columns = ["label"]
y_train_df.to_csv("./datasets/y_train.csv")
print("y_train: ", y_train)

X_train_df = pd.read_csv("./datasets_csv/pred_labeling/news/news_dataset.csv")
X_train_df.to_csv("./datasets/X_train.csv", index=False)

print("処理が完了しました。作成されたX_train,y_trainおよびattentionを確認してください。")
