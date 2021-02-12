import os
import pandas as pd

def make_table():
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
