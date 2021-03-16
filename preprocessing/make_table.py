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
    csv_path_news = ("./datasets/x_train.csv")

    # ニュース記事
    if not os.path.isfile(csv_path_news):
        print("NG:There is no news file.")
        exit
    else:
        df_news = pd.read_csv(csv_path_news)#, index_col=0
        print(df_news)
        print(df_news)

    # ニュース記事のラベル（自作）
    csv_path_label = ("./datasets/y_train.csv")

    # ニュース記事
    if not os.path.isfile(csv_path_label):
        print("NG:There is no label file.")
        exit
    else:
        df_label = pd.read_csv(csv_path_label, index_col=0)
        print(df_label)
        print(df_label)


    # データセット作成
    df = pd.concat([df_news, df_label], axis=1)
    print("df", df)


    # ソートは他のデータと結合する直前に行う（ラベルとずれてしまうため）
    df_s = df.sort_values('date')
    print("df_s", df_s)


    #一旦書き出し
    #df_s.to_csv('./datasets/x_train_sorted.csv')#, header=False, index=False



    # preprocessing_text 削除

    # tokenizer_mecab 削除

    # 関数移動　_get_indice

    # 単語分割する関数を定義

    # ベクトルをdfに
    # ID変換処理(今回はテキストのみ)
    #"""
    #TEXT_LEN=5000をどうするか。カウント関数利用を検討。
    #"""
    #TEXT_LEN=5000
    TEXT_LEN = preprocessing.get_max(df_s["text"])
    print("TEXT_LEN :", TEXT_LEN)

    df_news_temp = pd.DataFrame([])

    for i in range(len(df_s)):
        text = df_s.loc[i]["text"]
        pre_text = preprocessing.preprocessing_text(text)
        tokenized_text = preprocessing.tokenizer_mecab(pre_text)
        indice = preprocessing.get_indice(text , maxlen=TEXT_LEN)
        df_text_temp = pd.DataFrame(indice).T

        df_news_temp = pd.concat([df_news_temp, df_text_temp], ignore_index=True)#, axis = 1

    print("df_news_temp", df_news_temp)


    df_news_date = pd.DataFrame(df_s['date'], columns = ["date"])
    df_news_date = df_news_date.reset_index(drop=True)
    print("df_news_date", df_news_date)
    #df_news_date = df_s['date']
    #df_news_date.columns = ['date']
    #print("df_news_date", df_news_date)

    df_news_label = pd.DataFrame(df_s['label'], columns = ["label"])
    df_news_label = df_news_label.reset_index(drop=True)
    print("df_news_date", df_news_label)

    #print("-3", df_news_date)
    df_news_date['date'] = df_news_date["date"].apply(lambda w: pd.to_datetime(w))


    #print("-2", df_news_date)
    df_news_date = pd.concat([df_news_date, df_news_temp, df_news_label], axis = 1)#, ignore_index=True
    #df_news_date = pd.concat([df_news_date, df_news], axis = 1, ignore_index=True)#

    #print(type(df_news_date))
    #print(df_news_date.columns)
    #df_news_date.columns = df_news_date.columns.astype(str)
    #print(df_news_date.columns)
    #print("-1", df_news_date)
    # headerの変更
    #df_news_date.rename(columns={'0': 'date', '1': 'top'})
    #df_news_date.rename(columns={'0': 'date', '1': 'top'}, inplace=True)


    print("df_news_date.columns", df_news_date.columns)
    print("df_news_date", df_news_date)

    #一旦書き出し
    #df_news_date.to_csv('./datasets/x_train_ID.csv', index=False)#, header=False, index=False

    return df_news_date


def trend():
    csv_path_trend = ("./associated_data/multiTimeline/")

    df_trend = pd.DataFrame([])

    count = 0
    for file_name in os.listdir("./associated_data/multiTimeline"):
        if file_name.endswith(".csv"):
            df_temp = pd.read_csv(csv_path_trend + "/"+ file_name, header=1)#, header=None, header=0
            print(csv_path_trend + "/"+ file_name)
            print(df_temp)
            print(df_temp.columns)
            #print("df_temp.iloc[:, 1]", df_temp.iloc[:, 1])
            #df_trend = pd.concat([df_trend, df_temp.iloc[:, 1]], axis = 1)#, ignore_index=True
            if count == 0:
                df_trend = pd.concat([df_trend, df_temp], axis = 1)#, ignore_index=True
            else:
                df_trend = pd.concat([df_trend, df_temp.iloc[:, 1]], axis = 1)#, ignore_index=True
            count += 1

    #一旦書き出し
    print(df_trend)
    #df_trend.to_csv('./associated_data/dataframe_trend.csv', header=False)#, index=False

    #もう一度読み込み
    #df_trend_csv = pd.read_csv('./associated_data/dataframe_trend.csv')#, index_col=0
    #print(df_trend_csv[:]["avex: (日本)"])



    #print(df_trend_csv)
    #print(df_trend_csv.columns)


    #日付変換
    print(df_trend['週'])
    df_trend['週'] = df_trend["週"].apply(lambda w: pd.to_datetime(w))

    print(df_trend)

    #df_trend_csv.to_csv('./associated_data/dataframe_trend.csv', index=False)#, index=False

    return df_trend
    """
    #一旦書き出し
    print(df_trend)
    #df_trend.to_csv('./associated_data/dataframe_trend.csv', header=False)#, index=False

    #もう一度読み込み
    df_trend_csv = pd.read_csv('./associated_data/dataframe_trend.csv')#, index_col=0
    #print(df_trend_csv[:]["avex: (日本)"])



    print(df_trend_csv)
    #print(df_trend_csv.columns)


    #日付変換
    print(df_trend_csv['週'])
    df_trend_csv['週'] = df_trend_csv["週"].apply(lambda w: pd.to_datetime(w))

    print(df_trend_csv)

    #df_trend_csv.to_csv('./associated_data/dataframe_trend.csv', index=False)#, index=False

    return df_trend_csv
    """

def concat(df_trend, df_news):
    #日付だけのデータフレームを作成
    """
    データセット作成関数
    トレンドデータ、テキストデータと指標データを組み合わせて、
    欠損データの補いや欠損列の削除を行う
    （日付を自由に入れられるようにしたい。）
    """
    date_list = [datetime(2020, 1, 1, hour=0, minute=0, second=0) + timedelta(days=i) for i in range(397)]
    df_date = pd.DataFrame(date_list, columns = ["date"])
    #print(date_str_list)

    df_date['date'] = df_date["date"].apply(lambda w: pd.to_datetime(w))

    print(df_date)

    #trend_f_path = "./associated_data/dataframe_trend.csv"
    #df_trend = pd.read_csv(trend_f_path)#, index_col=0, header=False
    print(df_trend)

    #news_f_path = "./datasets/x_train_ID.csv"
    #df_news = pd.read_csv(news_f_path)#, index_col=0
    print("df_news", df_news)
    print("df_news.columns", df_news.columns)

    #sorted_f_path = "./datasets/x_train_sorted.csv"
    #df_s = pd.read_csv(sorted_f_path)#, index_col=0
    #print(df_s)

    index_f_path = "./associated_data/dataframe_indicator_index.csv"
    df_index_csv = pd.read_csv(index_f_path)#, index_col=0
    print(df_index_csv)


    #日付変換
    df_trend['週'] = df_trend["週"].apply(lambda w: pd.to_datetime(w))


    print(type(df_trend.iloc[0]["週"]))
    print(type(df_date.iloc[0]["date"]))


    # 結合用コード

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

    print(df_date.head(30))
    print(df_date.tail(30))

    #日付変換
    df_index_csv['date'] = df_index_csv["date"].apply(lambda w: pd.to_datetime(w))


    print(type(df_index_csv.iloc[0]["date"]))
    print(type(df_date.iloc[0]["date"]))


    # 結合用コード

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

    print(df_date.head(30))
    print(df_date.tail(30))

    # 数値データのサイズを取得
    n_index_feature = len(df_date.columns)
    print("n_index_feature", n_index_feature)

    #一旦書き出し
    #df_date.to_csv('./associated_data/dataframe_trend_and_index.csv')#, header=False, index=False


    # 欠損値だらけなので埋める

    # ''だとfillnaできないので置換する
    df_date = df_date.replace('', np.nan, regex=True)
    print("fillnaの前", df_date.tail(30))


    # 前の値で埋める
    df_date = df_date.fillna(method='ffill')
    print("fillna", df_date.tail(30))


    # テキストを結合する

    #日付変換
    #df_news['0'] = df_news['0'].apply(lambda w: pd.to_datetime(w))
    df_news['date'] = df_news['date'].apply(lambda w: pd.to_datetime(w))


    #print(type(df_news.iloc[0][0]))
    #print(type(df_date.iloc[0]["date"]))

    # 結合用コード
    # 単純にconcat
    print("単純にconcat")
    print("df_date", df_date.shape)
    print("df_date", df_date)
    print("df_news", df_news.shape)
    print("df_news", df_news)




    #IDベクトル
    columns_list = df_news.columns.values
    #columns_list = np.append(columns_list, 'label')

    print(columns_list)
    print(df_date.columns.values)

    text_tabel_len = len(df_news.columns.values)
    index_tabel_len = len(df_date.columns.values)
    print("text_tabel_len", text_tabel_len)
    print("index_tabel_len", index_tabel_len)
    #df_news

    for c in range(1, text_tabel_len):
        df_date[columns_list[c]] = ''
    #print(df_date)

    print("df_date ゼロ埋め", df_date.shape)
    print("df_date ゼロ埋め", df_date)

    for i in range(len(df_news)):
        d_y = df_news.loc[i]['date'].year
        d_m = df_news.loc[i]['date'].month
        d_d = df_news.loc[i]['date'].day
        #d_y = df_news.loc[i]['date'].dt.year#.astype(str)
        #d_m = df_news.loc[i]['date'].dt.month#.astype(str)
        #d_d = df_news.loc[i]['date'].dt.day#.astype(str)
        d_ymd = str(d_y)+'/'+str(d_m).zfill(2)+'/'+str(d_d).zfill(2)
        #d_ymd = str(d_y[0])+'/'+str(d_m[0]).zfill(2)+'/'+str(d_d[0]).zfill(2)
        date_t=pd.to_datetime(d_ymd).date()

        # +6は変更必要では？テーブルのサイズを取得していれる。
        for j in range(len(df_date)):
        #for j in range(30):
            if date_t == df_date.iloc[j]["date"]:
                for f in range(1, text_tabel_len):
                    ##if str(df_date.columns.values[f+index_tabel_len-1]) is not 'label':
                    #df_date.iloc[j, f+index_tabel_len-1] = df_news.iloc[i][int(df_date.columns.values[f+index_tabel_len-1])]
                    #    #print(df_date.iloc[j][f+7])
                    #    #print(df_news.iloc[i][int(df_date.columns.values[f+7])])

                    if str(df_date.columns.values[f+index_tabel_len-1]) == 'label':
                        df_date.loc[j, 'label'] = df_news.loc[i,'label']
                    else:
                        df_date.iloc[j, f+index_tabel_len-1] = df_news.iloc[i][int(df_date.columns.values[f+index_tabel_len-1])]
                        #print(df_date.iloc[j][f+7])
                        #print(df_news.iloc[i][int(df_date.columns.values[f+7])])

    #print(df_date.head(30))
    print(df_date.tail(100))

    #一旦書き出し
    #df_date.to_csv('./associated_data/dataframe_all.csv')#, header=False, index=False


    # ''だとfillnaできないので置換する
    df_date = df_date.replace('', np.nan, regex=True)
    print(df_date.tail(30))


    # 欠損地を含む行を削除する
    df_drop = df_date.dropna(how="any")
    print(df_drop)


    #一旦書き出し
    df_index_col = df_drop.iloc[:,1:n_index_feature]
    df_label_col = df_drop.loc[:,'label'].astype('int')

    df_drop_index = pd.concat([df_index_col,df_label_col], axis=1)
    #df_drop_index.to_csv('./associated_data/dataframe_all_index.csv', index=False)#, header=False, index=False

    df_drop_text = df_drop.astype('int')#.astype({'label':int})
    #df_drop_text.iloc[:,n_index_feature:].to_csv('./associated_data/dataframe_all_text.csv', index=False)#, header=False, index=False

    return df_drop_index, df_drop_text.iloc[:,n_index_feature:]
