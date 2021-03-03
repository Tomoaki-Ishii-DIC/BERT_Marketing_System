import pandas as pd
import os

def make_ds():
    #csv読み込み
    csv_path_fine_train = ("./datasets_csv/finetuning/train")
    csv_path_fine_test = ("./datasets_csv/finetuning/test")
    csv_path_pred_labeling = ("./datasets_csv/pred_labeling")

    # ニュース記事
    df_train_news = pd.read_csv(csv_path_pred_labeling + '/news/news_dataset.csv')

    #print(df_train_news)


    # ニュースコメント
    csv_folder = [csv_path_fine_train , csv_path_fine_test, csv_path_pred_labeling]

    file_count = 0
    for p in csv_folder:
        #file_count += sum((len(f) for _, _, f in os.walk(p + '/comments'))) - 1
        DIR = p + '/comments'
        file_count += len([name for name in os.listdir(DIR) if name[-4:] == '.csv'])


    for j, p in enumerate(csv_folder):
        cols = ['text', 'reply', 'good', 'bad']
        df_temp = pd.DataFrame(columns=cols)

        for i in range(file_count):
            n_file = str(i+1).zfill(3)
            file_name = "comment_dataset_" + n_file + ".csv"
            file_path = (p + "/comments/" + file_name)
            if not os.path.isfile(file_path):
                continue
            #print(str(i+1).zfill(3))#あとで消す
            df_cmt = pd.read_csv(file_path, index_col=0)
            #df_temp = pd.concat([df_temp, df_cmt], ignore_index=True)

            #代入
            if j <= 1:
                df_temp = pd.concat([df_temp, df_cmt], ignore_index=True)
            else:
                df_cmt.columns = ["feature", "reply", "good", "bad"]
                df_cmt[["feature", "good", "bad"]].to_csv("./datasets/pred_labeling/features_" + n_file + ".csv", index=False)
                #df_pred_comments = df_temp.copy()

        #代入
        if j == 0:
            df_fine_train_comments = df_temp.copy()
        elif j == 1:
            df_fine_test_comments = df_temp.copy()

    print("finetuning train:\n", df_fine_train_comments)
    print()
    print("finetuning test:\n", df_fine_test_comments)
    print()
    #print("pred labeling:\n", df_pred_comments)

    #ラベル
    csv_folder = [csv_path_fine_train , csv_path_fine_test]

    for j, p in enumerate(csv_folder):
        cols = ['label']
        df_temp = pd.DataFrame(columns=cols)

        for i in range(file_count):
            n_file = str(i+1).zfill(3)
            file_name = "comment_labels_" + n_file + ".csv"
            file_path = (p + "/labels/" + file_name)
            if not os.path.isfile(file_path):
                continue
            #print(str(i+1).zfill(3))#あとで消す
            df_labels = pd.read_csv(file_path)#, index_col=0
            # concat
            #print(df_labels)
            df_temp = pd.concat([df_temp, df_labels], ignore_index=True)

        #代入
        if j == 0:
            df_fine_train_labels = df_temp.copy()
        elif j == 1:
            df_fine_test_labels = df_temp.copy()

    print("labels train:\n", df_fine_train_labels)
    print()
    print("labels test:\n", df_fine_test_labels)


    # データセットの作成と保存
    datasets_folder =  ("./datasets/finetuning")

    #ファインチューニング　学習用
    df_fine_train_comments.columns = ["feature", "reply", "good", "bad"]
    df_fine_train_comments["feature"].to_csv(datasets_folder + "/train/features.csv", index=False)

    #ファインチューニング　テスト用
    df_fine_test_comments.columns = ["feature", "reply", "good", "bad"]
    df_fine_test_comments["feature"].to_csv(datasets_folder + "/test/features.csv", index=False)

    #ラベル付け-ネガポジ学習用
    df_fine_train_labels["label"].to_csv(datasets_folder + "/train/labels.csv", index=False)
    df_fine_test_labels["label"].to_csv(datasets_folder + "/test/labels.csv", index=False)
