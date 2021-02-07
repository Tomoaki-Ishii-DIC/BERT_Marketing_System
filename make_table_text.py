import os
import pandas as pd

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

# 単語分割する関数を定義
import MeCab
import re
import string
import sentencepiece as spm
import numpy as np


def preprocessing_text(text):
    '''
    前処理
    '''
    print(text)
    # 改行コードを消去
    text = re.sub('<br />', '', text)
    text = re.sub('\n', '', text)

    # カンマ、ピリオド以外の記号をスペースに置換
    for p in string.punctuation:
        if (p == ".") or (p == ","):
            continue
        else:
            text = text.replace(p, " ")

    for p in string.punctuation:
        if (p == "。") or (p == "、"):
            continue
        else:
            text = text.replace(p, "　")

    # ピリオドなどの前後にはスペースを入れておく
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    text = text.replace("。", " 。 ")
    text = text.replace("、", " 、 ")

    return text

#m_t = MeCab.Tagger('-Owakati -d /usr/lib/mecab/dic/mecab-ipadic-neologd')
#m_t = MeCab.Tagger('-Ochasen -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')
wakati = MeCab.Tagger("-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")

def tokenizer_mecab(text):
    '''
    分かち書き
    '''
    words = wakati.parse(text).split()

    return words



# ここでもsentence piece
sp = spm.SentencePieceProcessor()
sp.Load('./downloads/bert-wiki-ja/wiki-ja.model')


# 最大単語数分のID化された文を返す関数
# maxlenがなくてエラーになるので勝手に追加（maxlenは最大単語数か？）
def _get_indice(feature, maxlen):
#def _get_indice(feature):
    # インデックス ０で埋める
    indices = np.zeros((maxlen), dtype = np.int32)
    # 最初に[CLS]、最後に'[SEP]をつけてトークン作る
    tokens = []
    tokens.append('[CLS]')
    pre_text = preprocessing_text(feature)#追加
    tokenized_text = tokenizer_mecab(pre_text)#追加
    tokens.extend(tokenized_text)#追加
    #tokens.extend(sp.encode_as_pieces(feature))# sentence piece
    tokens.append('[SEP]')

    for t, token in enumerate(tokens):
        # 最大単語数までトークンの単語をindicesに入れていく
        if t >= maxlen:
            break
        try:
            indices[t] = sp.piece_to_id(token)# id化してくれる？
        except:
            logging.warn(f'{token} is unknown.')# コメントしてくれる
            indices[t] = sp.piece_to_id('<unk>')# id化してくれる？unknown

    # 最大単語数分のID化された文を返す
    return indices

# ベクトルをdfに
# ID変換処理(今回はテキストのみ)
"""
TEXT_LEN=5000をどうするか。カウント関数利用を検討。
"""
TEXT_LEN=5000
df_news = pd.DataFrame([])

for i in range(len(df_s)):
    text = df_s.loc[i]["text"]
    pre_text = preprocessing_text(text)
    tokenized_text = tokenizer_mecab(pre_text)
    indice = _get_indice(text , maxlen=TEXT_LEN)
    df_text_temp = pd.DataFrame(indice).T

    df_news = pd.concat([df_news, df_text_temp], ignore_index=True)#, axis = 1

print(df_news)


df_news_date = pd.DataFrame(df_s['date'], columns = ["date"])


df_news_date['date'] = df_news_date["date"].apply(lambda w: pd.to_datetime(w))


df_news_date = pd.concat([df_news_date, df_news], axis = 1, ignore_index=True)#


print(df_news_date)


#一旦書き出し
df_news_date.to_csv('./datasets/x_train_ID.csv', index=False)#, header=False, index=False
