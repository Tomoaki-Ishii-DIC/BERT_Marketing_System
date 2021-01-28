# BERT
BERT
# クイックスタート
1. テキスト集め
任意の場所にフォルダを作成  
Yahoo!ニュースの記事とコメントをコピーしてテキストファイルに貼り付けて保存  
以下の形式で保存すること  

news_text_001.txt  
news_text_002.txt  
news_text_003.txt  
comment_text_001.txt  
comment_text_002.txt  
comment_text_003.txt  

2. ファイルをフォルダに振り分ける
「datasets_text」フォルダ内に以下の様に格納する  
- finetuning
  - ファインチューニング用



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
