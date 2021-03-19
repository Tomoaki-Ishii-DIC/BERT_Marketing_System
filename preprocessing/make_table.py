import os
import pandas as pd
import numpy as np

from preprocessing import preprocessing

import MeCab
import re
import string
import sentencepiece as spm

from datetime import datetime, timedelta


def text():
    """
    他の数値データと組み合わせて利用するために
    テキストを時系列テーブルに変換する関数

    Parameters
    ----------------

    """
    csv_path_news = ("./datasets/x_train.csv")

    # ニュース記事
    if not os.path.isfile(csv_path_news):
        print("NG:There is no news file.")
        exit
    else:
        df_news = pd.read_csv(csv_path_news)#, index_col=0


    # ニュース記事のラベル（自作）
    csv_path_label = ("./datasets/y_train.csv")

    # ニュース記事
    if not os.path.isfile(csv_path_label):
        print("NG:There is no label file.")
        exit
    else:
        df_label = pd.read_csv(csv_path_label, index_col=0)


    # データセット作成
    df = pd.concat([df_news, df_label], axis=1)

    # ソートは他のデータと結合する直前に行う（ラベルとずれてしまうため）
    df_s = df.sort_values('date')

    TEXT_LEN = preprocessing.get_max(df_s["text"])

    df_news_temp = pd.DataFrame([])

    for i in range(len(df_s)):
        text = df_s.loc[i]["text"]
        pre_text = preprocessing.preprocessing_text(text)
        tokenized_text = preprocessing.tokenizer_mecab(pre_text)
        indice = preprocessing.get_indice(text , maxlen=TEXT_LEN)
        df_text_temp = pd.DataFrame(indice).T

        df_news_temp = pd.concat([df_news_temp, df_text_temp], ignore_index=True)#, axis = 1

    df_news_date = pd.DataFrame(df_s['date'], columns = ["date"])
    df_news_date = df_news_date.reset_index(drop=True)

    df_news_label = pd.DataFrame(df_s['label'], columns = ["label"])
    df_news_label = df_news_label.reset_index(drop=True)

    #日付型変換
    df_news_date['date'] = df_news_date["date"].apply(lambda w: pd.to_datetime(w))

    df_news_date = pd.concat([df_news_date, df_news_temp, df_news_label], axis = 1)#, ignore_index=True

    return df_news_date


def trend():
    """
    トレンドデータを組み合わせる関数

    Parameters
    ----------------

    """
    csv_path_trend = ("./associated_data/multiTimeline/")

    df_trend = pd.DataFrame([])

    count = 0
    for file_name in os.listdir("./associated_data/multiTimeline"):
        if file_name.endswith(".csv"):
            df_temp = pd.read_csv(csv_path_trend + "/"+ file_name, header=1)#, header=None, header=0
            print(csv_path_trend + "/"+ file_name)

            if count == 0:
                df_trend = pd.concat([df_trend, df_temp], axis = 1)#, ignore_index=True
            else:
                df_trend = pd.concat([df_trend, df_temp.iloc[:, 1]], axis = 1)#, ignore_index=True
            count += 1

    #日付型変換
    df_trend['週'] = df_trend["週"].apply(lambda w: pd.to_datetime(w))

    return df_trend


def concat(df_trend, df_news):
    """
    データセット作成関数
    トレンドデータ、テキストデータと指標データを組み合わせて、
    欠損データの補いや欠損列の削除を行う
    （日付を自由に入れられるようにしたい。）

    Parameters
    ----------------

    """
    # 重複している日を取得
    duplicate_date = []
    for s in range(1, len(df_news)):
        if df_news.loc[s]["date"].strftime('%Y/%m/%d') == df_news.loc[s-1]["date"].strftime('%Y/%m/%d'):
            duplicate_date.append(df_news.loc[s]["date"].strftime('%Y/%m/%d'))

    #開始終了日は変更できるようにした方がいい

    #2020/01/01から397日分の連続する日付を作成
    #(ニュースが同じ日に複数ある場合には重複させる)
    if len(duplicate_date) > 0:
        date_list = []
        for i in range(397):
            date_list.append(datetime(2020, 1, 1, hour=0, minute=0, second=0) + timedelta(days=i))
            for n_dup in range(len(duplicate_date)):
                if duplicate_date[n_dup]==pd.to_datetime(date_list[-1]).strftime('%Y/%m/%d'):
                    date_list.append(duplicate_date[n_dup])
    else:
        date_list = [datetime(2020, 1, 1, hour=0, minute=0, second=0) + timedelta(days=i) for i in range(397)]

    df_date = pd.DataFrame(date_list, columns = ["date"])
    df_date["date"] = df_date["date"].apply(lambda w: pd.to_datetime(w))


    index_f_path = "./associated_data/dataframe_indicator_index.csv"
    df_index_csv = pd.read_csv(index_f_path)#, index_col=0


    #日付型変換
    df_trend['週'] = df_trend["週"].apply(lambda w: pd.to_datetime(w))

    ########## 結合用コード ##########

    #Googleトレンド
    columns_list = df_trend.columns.values

    for c in range(1, len(columns_list)):
        df_date[columns_list[c]] = ''

    for i in range(len(df_trend)):
    #for i in range(100):
        for j in range(len(df_date)):
        #for j in range(30):
            if df_trend.iloc[i]["週"] == df_date.iloc[j]["date"]:
                for f in range(1, len(columns_list)):
                    df_date.loc[j, columns_list[f]] = df_trend.loc[i, columns_list[f]]

    #日付型変換
    df_index_csv['date'] = df_index_csv["date"].apply(lambda w: pd.to_datetime(w))


    #指標データ
    columns_list = df_index_csv.columns.values

    for c in range(1, len(columns_list)):
        df_date[columns_list[c]] = ''

    for i in range(len(df_index_csv)):
    #for i in range(100):
        for j in range(len(df_date)):
        #for j in range(30):
            if df_index_csv.iloc[i]["date"] == df_date.iloc[j]["date"]:
                for f in range(1, len(columns_list)):
                    df_date.loc[j, columns_list[f]] = df_index_csv.loc[i, columns_list[f]]

    # 数値データのサイズを取得
    n_index_feature = len(df_date.columns)

    # 前処理
    # 欠損値だらけなので埋める
    # ''だとfillnaできないので置換する
    df_date = df_date.replace('', np.nan, regex=True)

    # 前の値で埋める
    df_date = df_date.fillna(method='ffill')


    # テキストを結合する

    #日付型変換
    df_news['date'] = df_news['date'].apply(lambda w: pd.to_datetime(w))

    #IDベクトル
    columns_list = df_news.columns.values

    text_tabel_len = len(df_news.columns.values)
    index_tabel_len = len(df_date.columns.values)

    for c in range(1, text_tabel_len):
        df_date[columns_list[c]] = ''

    # 空白判定のためにnanで置き換える
    df_date = df_date.replace('', np.nan, regex=True)

    for i in range(len(df_news)):
        d_y = df_news.loc[i]['date'].year
        d_m = df_news.loc[i]['date'].month
        d_d = df_news.loc[i]['date'].day
        d_ymd = str(d_y)+'/'+str(d_m).zfill(2)+'/'+str(d_d).zfill(2)
        news_date_time = pd.to_datetime(d_ymd).date()

        for j in range(len(df_date)):
            if news_date_time == df_date.iloc[j]["date"] and pd.isnull(df_date.iloc[j]["label"]):
                for f in range(1, text_tabel_len):
                    if str(df_date.columns.values[f+index_tabel_len-1]) == 'label':
                        df_date.loc[j, 'label'] = df_news.loc[i,'label']
                    else:
                        df_date.iloc[j, f+index_tabel_len-1] = df_news.iloc[i][int(df_date.columns.values[f+index_tabel_len-1])]
                break#同一日付への重複入力を避けるため

    # 前処理
    # ''だとfillnaできないので置換する
    df_date = df_date.replace('', np.nan, regex=True)

    # 欠損地を含む行を削除する
    df_drop = df_date.dropna(how="any")


    df_index_col = df_drop.iloc[:,1:n_index_feature]
    df_label_col = df_drop.loc[:,'label'].astype('int')
    df_drop_index = pd.concat([df_index_col,df_label_col], axis=1)

    df_drop_text = df_drop.astype('int')#.astype({'label':int})

    df_drop_index = df_drop_index.reset_index(drop=True)
    df_drop_text = df_drop_text.iloc[:,n_index_feature:].reset_index(drop=True)

    return df_drop_index, df_drop_text
