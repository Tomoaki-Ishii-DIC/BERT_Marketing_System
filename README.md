# BERT
BERT Marketing System
# クイックスタート
1. テキスト集め

任意の場所にフォルダを作成  
Yahoo!ニュースの記事とコメントをコピーしてテキストファイルに貼り付けて保存  
以下の形式で保存すること（番号は001から初めて通し番号をつける）  

news_text_001.txt  
news_text_002.txt  
news_text_003.txt  
comment_text_001.txt  
comment_text_002.txt  
comment_text_003.txt  

2. ファイルをフォルダに振り分ける

「datasets_text」フォルダ内に以下の様に振り分ける（ファイル名は変更しない）  
- finetuning
  - BERTのファインチューニング用
- pred_labelingfinetuning
  - LSTMの学習用（ラベリングはBERTが行う）

datasets_text  
├── finetuning  
│   ├── test  
│   │   ├── comments  
│   │   │   ├── comment_text_xxx.txt  
│   │   │   └── comment_text_xxx.txt  
│   │   └── news  
│   │       ├── news_text_xxx.txt  
│   │       └── news_text_xxx.txt  
│   └── train  
│       ├── comments  
│       │   ├── comment_text_xxx.txt  
│       │   └── comment_text_xxx.txt  
│       └── news  
│           ├── news_text_xxx.txt  
│           └── news_text_xxx.txt  
└── pred_labeling  
    ├── comments  
    │   ├── comment_text_xxx.txt  
    │   └── comment_text_xxx.txt  
    └── news  
        ├── news_text_xxx.txt  
        └── news_text_xxx.txt  

3. プログラムで表を作成（csv出力）

以下のファイルを実行
- make_news_csv.py
- make_comments_csv.py

「datasets_csv」フォルダにファイルが作成されたことを確認する

4. ファインチューニング用のラベルを作成する

「/datasets_csv/finetuning/test/comments」内のファイルを開きそれぞれのコメントに対応するラベルをつける。  
ファイル名は以下の形式に従う。

comment_labels_001.csv  
comment_labels_002.csv  
comment_labels_003.csv  

|  label  |
| ---- |
|  negative  |
|  positive  |
|  negative  |
|  positive  |

5. 日本語学習済みモデルをダウンロードする

以下のモデルを利用しているため、以下のURLにアクセスしGoogleドライブへのリンクから「bert-wiki-ja」フォルダをダウンロードして「downloads」フォルダへ入れる。

https://yoheikikuta.github.io/bert-japanese/


6. 設定ファイルを変更

「/downloads/bert-wiki-ja_config」内の’bert_finetuning_config_v1.json’を開く。  

最大単語数を必要に応じて変更する。  

"max_position_embeddings": 300,  
"max_seq_length": 300,  

最大単語数は「Sprint26_卒業課題_Keras_BERT_AWS.ipynb」、「Sprint26_卒業課題_Keras_BERT_local.ipynb」ファイル内の変数’max_token_num’として出力されるので、実行中に変更が必要になればその都度変更する。


７. 「Sprint26_卒業課題_Keras_BERT_AWS.ipynb」を開いて処理を実行する

全ての処理が完了した後、「/datasets」フォルダ内に’y_train.csv’が作成されていることを確認する。

８. 「Sprint26_卒業課題_Keras_BERT_local.ipynb」を開いて処理を実行する

作業完了
