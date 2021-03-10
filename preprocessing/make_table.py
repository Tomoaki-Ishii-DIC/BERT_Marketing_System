import os
import pandas as pd

from preprocessing import preprocessing

import MeCab
import re
import string
import sentencepiece as spm
import numpy as np

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
    print(df)


    # ソートは他のデータと結合する直前に行う（ラベルとずれてしまうため）
    df_s = df.sort_values('date')
    print(df_s)


    #一旦書き出し
    df_s.to_csv('./datasets/x_train_sorted.csv')#, header=False, index=False



    # preprocessing_text 削除

    # tokenizer_mecab 削除

    # 関数移動　_get_indice

    # 単語分割する関数を定義

    # ベクトルをdfに
    # ID変換処理(今回はテキストのみ)
    """
    TEXT_LEN=5000をどうするか。カウント関数利用を検討。
    """
    #TEXT_LEN=5000
    TEXT_LEN = preprocessing.get_max(df_s["text"])
    print("TEXT_LEN :", TEXT_LEN)

    df_news = pd.DataFrame([])

    for i in range(len(df_s)):
        text = df_s.loc[i]["text"]
        pre_text = preprocessing.preprocessing_text(text)
        tokenized_text = preprocessing.tokenizer_mecab(pre_text)
        indice = preprocessing.get_indice(text , maxlen=TEXT_LEN)
        df_text_temp = pd.DataFrame(indice).T

        df_news = pd.concat([df_news, df_text_temp], ignore_index=True)#, axis = 1

    print(df_news)


    df_news_date = pd.DataFrame(df_s['date'], columns = ["date"])


    df_news_date['date'] = df_news_date["date"].apply(lambda w: pd.to_datetime(w))


    df_news_date = pd.concat([df_news_date, df_news], axis = 1, ignore_index=True)#


    print(df_news_date)


    #一旦書き出し
    df_news_date.to_csv('./datasets/x_train_ID.csv', index=False)#, header=False, index=False


def table():
    csv_path_trend = ("./associated_data/multiTimeline/")

    count = 0
    for file_name in os.listdir("./associated_data/multiTimeline"):
        if file_name.endswith(".csv"):
            df_temp = pd.read_csv(csv_path_trend + "/"+ file_name)#, header=None, header=0
            print(csv_path_trend + "/"+ file_name)
            print(df_temp)
            print(df_temp.columns)
            if count == 0:
                df_trend = df_temp
            else:
                df_trend = pd.concat([df_trend, df_temp], axis = 1, ignore_index=True)
            count += 1

    #一旦書き出し
    print(df_trend)
    df_trend.to_csv('./associated_data/dataframe_trend.csv', header=False)#, index=False

    #もう一度読み込み
    df_trend_csv = pd.read_csv('./associated_data/dataframe_trend.csv')#, index_col=0
    #print(df_trend_csv[:]["avex: (日本)"])



    print(df_trend_csv)
    #print(df_trend_csv.columns)


    #日付変換
    print(df_trend_csv['週'])
    df_trend_csv['週'] = df_trend_csv["週"].apply(lambda w: pd.to_datetime(w))

    print(df_trend_csv)

    df_trend_csv.to_csv('./associated_data/dataframe_trend.csv', index=False)#, index=False
