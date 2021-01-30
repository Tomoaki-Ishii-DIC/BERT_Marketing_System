# BERT Marketing System

This is a marketing model using BERT.

# クイックスタート

1. テキスト集め


    Yahoo!ニュースの記事とコメントをコピーしてテキストファイルに貼り付けて保存(任意のフォルダへ)

    以下の形式で保存すること（番号は001から始まる通し番号にする。）  

    news_text_001.txt  
    news_text_002.txt  
    news_text_003.txt  
    comment_text_001.txt  
    comment_text_002.txt  
    comment_text_003.txt  

2. ファイルをフォルダに振り分ける

    ./datasets_text フォルダ内に以下の様に振り分ける（ファイル名は変更しない）  

    - finetuning
      - BERTのファインチューニング用
    - pred_labelingfinetuning
      - LSTMの学習用（ラベリングはBERTが行う）


    ./datasets_text  
    &emsp;└─ test  
    &emsp;&emsp;└─ comments  
    &emsp;&emsp;&emsp;└─ comment_text_xxx.txt  
    &emsp;&emsp;&emsp;└─ comment_text_xxx.txt  
    &emsp;&emsp;└─ news  
    &emsp;&emsp;&emsp;└─ news_text_xxx.txt  
    &emsp;&emsp;&emsp;└─ news_text_xxx.txt  
    &emsp;└─ train  
    &emsp;&emsp;└─ comments  
    &emsp;&emsp;&emsp;└─ comment_text_xxx.txt  
    &emsp;&emsp;&emsp;└─ comment_text_xxx.txt  
    &emsp;&emsp;└─ news  
    &emsp;&emsp;&emsp;└─ news_text_xxx.txt  
    &emsp;&emsp;&emsp;└─ news_text_xxx.txt  
    &emsp;└─ pred_labeling  
    &emsp;&emsp;└─ comments  
    &emsp;&emsp;&emsp;└─ comment_text_xxx.txt  
    &emsp;&emsp;&emsp;└─ comment_text_xxx.txt  
    &emsp;&emsp;└─ news  
    &emsp;&emsp;&emsp;└─ news_text_xxx.txt  
    &emsp;&emsp;&emsp;└─ news_text_xxx.txt  

3. プログラムで表を作成（csv出力）

    以下のファイルを実行
    - make_news_csv.py
    - make_comments_csv.py

    「datasets_csv」フォルダにファイルが作成されたことを確認する

4. ファインチューニング用のラベルを作成する

    ./datasets_csv/finetuning/test/comments 内のファイルを開きそれぞれのコメントに対応するラベルをつける。  
    ファイル名は以下の形式に従う。

    comment_labels_001.csv  
    comment_labels_002.csv  
    comment_labels_003.csv  

    ファイルの中は以下の形式で記載する。

    |  label  |
    | ---- |
    |  positive  |
    |  negative  |
    |  positive  |
    |  negative  |
    |  positive  |
    |  negative  |

5. 日本語学習済みモデルをダウンロードする

    以下のモデルを利用しているため、以下のURLにアクセスしGoogleドライブへのリンクから「bert-wiki-ja」フォルダをダウンロードして ./downloads フォルダへ入れる。

    https://yoheikikuta.github.io/bert-japanese/


6. 設定ファイルを変更

    ./downloads/bert-wiki-ja_config 内の’bert_finetuning_config_v1.json’を開く。  

    最大単語数を必要に応じて変更する。  

    >"max_position_embeddings": 300,  
    >"max_seq_length": 300,  

    最大単語数は「Sprint26_卒業課題_Keras_BERT_AWS.ipynb」、「Sprint26_卒業課題_Keras_BERT_local.ipynb」ファイル内の変数’max_token_num’として出力されるので、実行中に変更が必要になればその都度変更する。


７. BERTのNotebookを実行

    用途に応じて以下のいづれかのファイルを使用  

    - Sprint26_卒業課題_Keras_BERT_AWS.ipynb
    - Sprint26_卒業課題_Keras_BERT_local.ipynb

    全ての処理が完了した後、 ./datasets フォルダ内に y_train.csv が作成されていることを確認する。

8. Attention（キーワード）の確認

    ./attention_excel フォルダに書くニュース記事に対応した'.xlxs'ファイルが作成されるので、開いて中身を確認する。  
    Self-Attentionが高い言葉がネガポジ判定に寄与した単語と考えられるため、文脈からキーワードを探し出す。  

    attention_001.xlsx  
    attention_002.xlsx  
    attention_003.xlsx  

9. Googleトレンドデータの取得

    - Googleトレンドデータの取得
    Googleトレンドで上記のキーワードのトレンドを一つずつ表示し、csvにてダウンロードして ./associated_data/multiTimeline に保管する。  

    - その他の指標のデータ
    任意のデータを取得して ./associated_data/multiTimeline に保管する  
    (「Sprint26_卒業課題_Keras_RNN.ipynb」に読み込みやテーブル化するコードを追加してください。)


10. RNNのNotebookを実行する

    以下のファイルを実行する

    - Sprint26_卒業課題_Keras_RNN.ipynb

    ファイル内のネガポジ判定結果を確認する。


    作業完了
