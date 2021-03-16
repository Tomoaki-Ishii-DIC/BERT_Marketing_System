import pandas as pd
import os

def make_ds():
    """
    データセットを作成する関数

    Parameters
    ----------------

    """
    #csv読み込み
    csv_path_fine_train = ("./datasets_csv/finetuning/train")
    csv_path_fine_test = ("./datasets_csv/finetuning/test")
    csv_path_pred_labeling = ("./datasets_csv/pred_labeling")


    # ニュースコメントパスのリスト
    csv_folder = [csv_path_fine_train , csv_path_fine_test, csv_path_pred_labeling]

    # ニュースコメントのファイル数を取得
    file_count = 0
    for p in csv_folder:
        DIR = p + '/comments'
        file_count += len([name for name in os.listdir(DIR) if name[-4:] == '.csv'])

    # ニュースコメントのデータセットを作成
    for j, p in enumerate(csv_folder[:2]):
        cols = ['text', 'reply', 'good', 'bad']
        df_temp = pd.DataFrame(columns=cols)

        for i in range(file_count):
            n_file = str(i+1).zfill(3)
            file_name = "comment_dataset_" + n_file + ".csv"
            file_path = (p + "/comments/" + file_name)
            if not os.path.isfile(file_path):
                continue
            df_cmt = pd.read_csv(file_path, index_col=0)
            #結合
            df_temp = pd.concat([df_temp, df_cmt], ignore_index=True)

        #代入
        if j == 0:
            df_fine_train_comments = df_temp.copy()
        elif j == 1:
            df_fine_test_comments = df_temp.copy()

    #print("finetuning train:\n", df_fine_train_comments)
    #print("finetuning test:\n", df_fine_test_comments)

    # 各ニュースコメントに対するラベルを作成
    for j, p in enumerate(csv_folder):
        cols = ['label']
        df_temp = pd.DataFrame(columns=cols)

        for i in range(file_count):
            n_file = str(i+1).zfill(3)
            file_name = "comment_labels_" + n_file + ".csv"
            file_path = (p + "/labels/" + file_name)
            if not os.path.isfile(file_path):
                continue
            df_labels = pd.read_csv(file_path)#, index_col=0
            #結合
            df_temp = pd.concat([df_temp, df_labels], ignore_index=True)

        #代入
        if j == 0:
            df_fine_train_labels = df_temp.copy()
        elif j == 1:
            df_fine_test_labels = df_temp.copy()

    #print("labels train:\n", df_fine_train_labels)
    #print("labels test:\n", df_fine_test_labels)

    # データセットの出力
    datasets_folder =  ("./datasets/finetuning")

    #ファインチューニング　学習用
    df_fine_train_comments.columns = ["feature", "reply", "good", "bad"]
    df_train_features = df_fine_train_comments["feature"]

    #ファインチューニング　テスト用
    df_fine_test_comments.columns = ["feature", "reply", "good", "bad"]
    df_test_features = df_fine_test_comments["feature"]

    #ラベル付け-ネガポジ学習用
    df_train_labels = df_fine_train_labels["label"]
    df_test_labels = df_fine_test_labels["label"]

    return df_train_features, df_test_features, df_train_labels, df_test_labels
