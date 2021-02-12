from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def concat():
    #日付だけのデータフレームを作成
    """
    日付を自由に入れられるようにする。
    """
    date_list = [datetime(2020, 1, 1, hour=0, minute=0, second=0) + timedelta(days=i) for i in range(397)]
    df_date = pd.DataFrame(date_list, columns = ["date"])
    #print(date_str_list)



    df_date['date'] = df_date["date"].apply(lambda w: pd.to_datetime(w))

    print(df_date)


    trend_f_path = "./associated_data/dataframe_trend.csv"
    df_trend_csv = pd.read_csv(trend_f_path)#, index_col=0, header=False
    print(df_trend_csv)

    news_f_path = "./datasets/x_train_ID.csv"
    df_news_csv = pd.read_csv(news_f_path)#, index_col=0
    print(df_news_csv)

    sorted_f_path = "./datasets/x_train_sorted.csv"
    df_s = pd.read_csv(sorted_f_path)#, index_col=0
    print(df_s)

    index_f_path = "./associated_data/dataframe_indicator_index.csv"
    df_index_csv = pd.read_csv(index_f_path)#, index_col=0
    print(df_index_csv)


    #日付変換
    df_trend_csv['週'] = df_trend_csv["週"].apply(lambda w: pd.to_datetime(w))


    print(type(df_trend_csv.iloc[0]["週"]))
    print(type(df_date.iloc[0]["date"]))


    # 結合用コード

    #Googleトレンド
    columns_list = df_trend_csv.columns.values

    for c in range(1, len(columns_list)):
        df_date[columns_list[c]] = ''

    for i in range(len(df_trend_csv)):
    #for i in range(100):
        for j in range(len(df_date)):
        #for j in range(30):
            if df_trend_csv.iloc[i]["週"] == df_date.iloc[j]["date"]:
                for f in range(1, len(columns_list)):
                    df_date.loc[j, columns_list[f]] = df_trend_csv.loc[i, columns_list[f]]

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


    #一旦書き出し
    df_date.to_csv('./associated_data/dataframe_trend_and_index.csv')#, header=False, index=False


    # 欠損値だらけなので埋める

    # ''だとfillnaできないので置換する
    df_date = df_date.replace('', np.nan, regex=True)
    print("fillnaの前", df_date.tail(30))


    # 前の値で埋める
    df_date = df_date.fillna(method='ffill')
    print("fillna", df_date.tail(30))


    # テキストを結合する

    #日付変換
    df_news_csv['0'] = df_news_csv['0'].apply(lambda w: pd.to_datetime(w))


    #print(type(df_news_csv.iloc[0][0]))
    #print(type(df_date.iloc[0]["date"]))

    # 結合用コード

    #IDベクトル
    columns_list = df_news_csv.columns.values
    columns_list = np.append(columns_list, 'label')

    print(columns_list)
    print(df_date.columns.values)
    df_news_csv

    for c in range(1, len(columns_list)):
        df_date[columns_list[c]] = ''
    #print(df_date)

    for i in range(len(df_news_csv)):
    #for i in range(100):
        d_y = df_news_csv.iloc[i][0].year
        d_m = df_news_csv.iloc[i][0].month
        d_d = df_news_csv.iloc[i][0].day
        d_ymd = str(d_y)+'/'+str(d_m)+'/'+str(d_d)
        date_t=pd.to_datetime(d_ymd).date()
        for j in range(len(df_date)):
        #for j in range(30):
            if date_t == df_date.iloc[j]["date"]:
                for f in range(1, len(columns_list)):
                    if str(df_date.columns.values[f+6]) == 'label':
                        df_date.loc[j, 'label'] = df_s.loc[i,'label']
                    else:
                        df_date.iloc[j, f+6] = df_news_csv.iloc[i][int(df_date.columns.values[f+6])]
                        #print(df_date.iloc[j][f+7])
                        #print(df_news_csv.iloc[i][int(df_date.columns.values[f+7])])

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
    df_drop.to_csv('./associated_data/dataframe_all.csv')#, header=False, index=False
