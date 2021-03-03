import os
import re
from datetime import datetime, date
import pandas as pd


def make_cmt(input_path, output_path):
    """
    ニュースコメントが記載されたテキストファイルを整形してcsvで保存する関数

    Parameters
    ----------------
    input_path : list, shape(３,)
        ニュースコメントが記載されたテキストファイルのパス
    output_path : list, shape(３,)
        データセット（csv）の保存先パス
    """
    # テキストファイルの数を取得
    file_count = 0
    for p in range(3):
        #file_count += sum((len(f) for _, _, f in os.walk(input_path[p])))
        file_count += len([name for name in os.listdir(input_path[p]) if name[-4:] == '.txt'])

    # データセット作成
    for p in range(3):
        converted_files_count = 0
        unconverted_files_count = 0
        saved_csv_count = 0

        for i in range(file_count):
            # ゼロ埋め
            n_file = str(i+1).zfill(3)
            file_name = "comment_text_" + n_file + ".txt"
            f_path = (input_path[p] + "/" + file_name)
            if not os.path.isfile(f_path):
                unconverted_files_count += 1
                continue
            data_set = []
            with open(f_path, 'r', encoding='UTF-8') as f:
                sentence_judg = False
                while True:
                    line = f.readline()
                    if not line:
                        break

                    # フラグ
                    if '|' in line:
                        feature = 0#column count
                    elif '返信' in line:
                        feature = 2#column count
                        sentence_judg = False

                    # 処理
                    if feature == 0:
                        data_list = []
                        data_list = data_list + line.split('|')
                        sentence_judg = True
                        feature = 1#column count
                        sentence = ''
                    elif sentence_judg == True:
                        sentence += line
                    elif feature == 2:
                        data_list.append(sentence)
                        data_list.append(re.sub(r"[返信\n]", "", line))
                        feature += 1
                    elif feature > 2:
                        if not line == '\n':
                            data_list.append(line.replace('\n', ''))
                            feature += 1

                    if feature == 5:
                         data_set.append(data_list)

                converted_files_count += 1

            # 保存処理
            df = pd.DataFrame(data_set, columns=['id', 'str_date', 'text', 'reply', 'good', 'bad'])
            save_csv_name = output_path[p] + "/" + "comment_dataset_" + n_file + ".csv"

            if not os.path.exists(save_csv_name):
                df[['text', 'reply', 'good', 'bad']].to_csv(save_csv_name)
                saved_csv_count += 1

        # メッセージ出力
        not_overwritten = converted_files_count - saved_csv_count
        if not_overwritten > 0:
            print("{} に{}個のcsvファイルが作成されました。{}個のファイルが上書きされませんでした。"\
                .format(output_path[p], saved_csv_count, not_overwritten))
        else:
            print("{} に{}個のcsvファイルが作成されました。"\
                .format(output_path[p], saved_csv_count))

        unknown = file_count-(converted_files_count+unconverted_files_count)
        if unknown > 0:
            print("{}個のファイルについては実行結果が不明です。")


def make_news(input_path, output_path, output_file_name):
    """
    ニュース記事が記載されたテキストファイルを整形してcsvで保存する関数

    Parameters
    ----------------
    input_path : list, shape(３,)
        ニュース記事が記載されたテキストファイルのパス
    output_path : list, shape(３,)
        データセット（csv）の保存先パス
    output_file_name : list, shape(３,)
        作成したcsvの保存ファイル名
    """
    # テキストファイルの数を取得
    file_count = 0
    for p in range(3):
        #file_count += sum((len(f) for _, _, f in os.walk(input_path[p])))
        file_count += len([name for name in os.listdir(input_path[p]) if name[-4:] == '.txt'])

    # データセット作成
    for p in range(3):
        converted_files_count = 0
        unconverted_files_count = 0
        saved_csv_count = 0

        data_set = []
        for i in range(file_count):
            n_file = str(i+1).zfill(3)
            file_name = "news_text_" + n_file + ".txt"
            f_path = (input_path[p] + "/" + file_name)
            if not os.path.isfile(f_path):
                unconverted_files_count += 1
                continue
            with open(f_path, 'r', encoding='UTF-8') as f:
                count = 0
                while True:
                    line = f.readline()
                    if not line:
                        data_list.append(sentence)
                        data_set.append(data_list)
                        break

                    # 処理
                    if count == 0:
                        data_list = []
                        data_list.append(line)
                        sentence = ''
                    elif count == 1:
                        data_list.append(line)
                    elif count >= 2:
                        if not line == '\n':
                            sentence += line

                    count += 1

                converted_files_count += 1

        df = pd.DataFrame(data_set, columns=['title', 'str_date', 'text'])

        date_list = []
        for a in range(len(df)):
            date_text = df.loc[a]['str_date']
            # '('、')'、'配'のいづれかで分割
            split_text = re.split('[()配]',date_text)
            t_datetime = split_text[0].strip().replace('/', '-') + ' ' + split_text[2].strip() + ':00'
            if t_datetime.count('-') == 1:
                this_year = str(date.today().year)
                t_datetime = this_year + '-' + t_datetime
            dt_datetime = datetime.strptime(t_datetime, '%Y-%m-%d %H:%M:%S')
            date_list.append(dt_datetime)

        # 保存処理
        df.insert(0, 'date', date_list)
        # ソートはここでは行わず結合する直前に行う（ラベルとずれてしまうため）
        #df_s = df.sort_values('date')

        save_csv_name = output_path[p] + "/" + output_file_name[p]
        if not os.path.exists(save_csv_name):
            df[['date', 'title', 'text']].to_csv(save_csv_name, index=False)
            saved_csv_count += 1

        # メッセージ出力
        not_overwritten = 1 - saved_csv_count
        if not_overwritten > 0:
            print("{} に{}個のcsvファイルが作成されました。{}個のファイルが上書きされませんでした。"\
                .format(output_path[p], saved_csv_count, not_overwritten))
        else:
            print("{} に{}個のcsvファイルが作成されました。"\
                .format(output_path[p], saved_csv_count))

        unknown = file_count-(converted_files_count+unconverted_files_count)
        if unknown > 0:
            print("{}個のファイルについては実行結果が不明です。")



# 読み込むテキストファイルとcsvの出力先を指定（train、test、pred_labeling）
input_path_cmt = [("./datasets_text/finetuning/train/comments"),
                ("./datasets_text/finetuning/test/comments"),
                ("./datasets_text/pred_labeling/comments")]
output_path_cmt = [("./datasets_csv/finetuning/train/comments"),
                ("./datasets_csv/finetuning/test/comments"),
                ("./datasets_csv/pred_labeling/comments")]

# コメントのデータセット作成
make_cmt(input_path_cmt, output_path_cmt)

# 読み込むテキストファイルとcsvの出力先を指定（train、test、pred_labeling）
input_path_news = [("./datasets_text/finetuning/train/news"),
                ("./datasets_text/finetuning/test/news"),
                ("./datasets_text/pred_labeling/news")]
output_path_news = [("./datasets_csv/finetuning/train/news"),
                ("./datasets_csv/finetuning/test/news"),
                ("./datasets_csv/pred_labeling/news")]

# 出力ファイル名を指定（train、test、pred_labeling）
output_file_name = ["news_dataset_train.csv",
                    "news_dataset_test.csv",
                    "news_dataset.csv"]

# ニュース記事のデータセット作成
make_news(input_path_news, output_path_news, output_file_name)

print("csvファイルの作成が完了しました。ファインチューニング用のラベルを作成してください。")
