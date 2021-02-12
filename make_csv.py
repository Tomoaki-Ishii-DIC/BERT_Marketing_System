import pandas as pd
import os
import re
from datetime import datetime as dt
import datetime

def make_cmt():
    folder_path = [("./datasets_text/finetuning/train/comments"),
                    ("./datasets_text/finetuning/test/comments"),
                    ("./datasets_text/pred_labeling/comments")]
    folder_path_output = [("./datasets_csv/finetuning/train/comments"),
                    ("./datasets_csv/finetuning/test/comments"),
                    ("./datasets_csv/pred_labeling/comments")]

    file_count = 0
    for p in range(3):
        file_count += sum((len(f) for _, _, f in os.walk(folder_path[p])))

    for p in range(3):
        #file_count = sum((len(f) for _, _, f in os.walk(folder_path[p])))

        for i in range(file_count):
            print(str(i+1).zfill(3))
            n_file = str(i+1).zfill(3)
            file_name = "comment_text_" + n_file + ".txt"
            f_path = (folder_path[p] + "/" + file_name)
            if not os.path.isfile(f_path):
                continue
            data_set = []
            with open(f_path, 'r', encoding='UTF-8') as f:
                sentence_judg = False
                while True:
                    line = f.readline()
                    if not line:
                        break

                    # flag
                    if '|' in line:
                        feature = 0#column count
                    elif '返信' in line:
                        feature = 2#column count
                        sentence_judg = False

                    # processing
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
                        data_list.append(re.sub(r"[返信\n]", "", line))#.replace('返信', '')
                        feature += 1
                    elif feature > 2:
                        if not line == '\n':
                            data_list.append(line.replace('\n', ''))#.replace('\n', '')
                            feature += 1

                    if feature == 5:
                         data_set.append(data_list)

            df = pd.DataFrame(data_set, columns=['id', 'str_date', 'text', 'reply', 'good', 'bad'])
            print(df[['text', 'reply', 'good', 'bad']])
            print()
            df[['text', 'reply', 'good', 'bad']].to_csv(folder_path_output[p] + "/" + "comment_dataset_" + n_file + ".csv")
            #print(df[['id', 'text', 'reply', 'good', 'bad']])
            #print()
            #df[['id', 'text', 'reply', 'good', 'bad']].to_csv(folder_path_output[p] + "/" + "comment_dataset_" + n_file + ".csv")




def make_news():
    folder_path = [("./datasets_text/finetuning/test/news"),
                    ("./datasets_text/finetuning/train/news"),
                    ("./datasets_text/pred_labeling/news")]
    folder_path_output = [("./datasets_csv/finetuning/news"),
                    ("./datasets_csv/finetuning/news"),
                    ("./datasets_csv/pred_labeling/news")]
    folder_path_x_train = ("./datasets")

    file_count = 0
    for p in range(3):
        file_count += sum((len(f) for _, _, f in os.walk(folder_path[p])))

    for p in range(3):
        #file_count = sum((len(f) for _, _, f in os.walk(folder_path[p])))

        data_set = []
        for i in range(file_count):
            n_file = str(i+1).zfill(3)
            file_name = "news_text_" + n_file + ".txt"
            f_path = (folder_path[p] + "/" + file_name)
            if not os.path.isfile(f_path):
                continue
            with open(f_path, 'r', encoding='UTF-8') as f:
                count = 0
                while True:
                    line = f.readline()
                    if not line:
                        data_list.append(sentence)
                        data_set.append(data_list)
                        break

                    # processing
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

        df = pd.DataFrame(data_set, columns=['title', 'str_date', 'text'])

        date_list = []
        for a in range(len(df)):
            date_text = df.loc[a]['str_date']
            # '('、')'、'配'のいづれかで分割
            split_text = re.split('[()配]',date_text)
            t_datetime = split_text[0].strip().replace('/', '-') + ' ' + split_text[2].strip() + ':00'
            if t_datetime.count('-') == 1:
                this_year = str(datetime.date.today().year)
                t_datetime = this_year + '-' + t_datetime
            dt_datetime = dt.strptime(t_datetime, '%Y-%m-%d %H:%M:%S')
            date_list.append(dt_datetime)

        df.insert(0, 'date', date_list)
        # ソートは結合する直前に行う（ラベルとずれてしまうため）
        #df_s = df.sort_values('date')
        #print(df_s[['date', 'title', 'text']])
        print(df[['date', 'title', 'text']])
        print()

        if p == 2:
            df[['date', 'title', 'text']].to_csv(folder_path_output[p] + "/" + "news_dataset.csv", index=False)
            df[['date', 'title', 'text']].to_csv(folder_path_x_train + "/" + "x_train.csv", index=False)

# 処理
make_cmt()
make_news()
