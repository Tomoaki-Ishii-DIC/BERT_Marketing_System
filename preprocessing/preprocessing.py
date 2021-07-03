import MeCab
import re
import os
import subprocess
import string
import sentencepiece as spm
import numpy as np

from transformers import BertJapaneseTokenizer
from tensorflow import keras

def preprocessing_text(text):
    """
    前処理用の関数。改行の削除等を行う。

    Parameters
    ----------------
    text : str
        ニュースコメントのテキスト
    """
    # 改行コードを消去
    text = re.sub('<br />', '', text)
    text = re.sub('\n', '', text)

    # カンマ、ピリオド以外の記号をスペースに置換
    for p in string.punctuation:
        if (p == ".") or (p == ",") or (p == "。") or (p == "、"):
            continue
        else:
            text = text.replace(p, " ")

    # ピリオドなどの前後にはスペースを入れておく
    text = text.replace(".", " . ")
    text = text.replace(",", " , ")
    text = text.replace("。", " 。 ")
    text = text.replace("、", " 、 ")

    return text


def tokenizer_mecab(text):
    """
    テキストを分かち書きに変換する関数

    Parameters
    ----------------
    text : str
        ニュースコメントのテキスト
    wakati : Mecab Instance
        Mecabの分かち書きインスタンス
    """
    cmd = 'echo `mecab-config --dicdir`'
    neologd_lib = (subprocess.Popen(cmd, stdout=subprocess.PIPE,
                           shell=True).communicate()[0]).decode('utf-8').replace( '\n' , '' )
    wakati = MeCab.Tagger('-Owakati -d' + ' ' + neologd_lib + '/mecab-ipadic-neologd')

    words = wakati.parse(text).split()

    return words


def get_max(X):
    """
    最大トークン数を取得。複数の文から最大のトークン数を取得する。

    Parameters
    ----------------
    X : str, dataframe shape(n_samples,1)
        ニュースコメントをまとめたデータフレーム
    """
    numbers = []
    for feature in X:
        #features_number = get_numbers_indice(feature)
        tokens = []
        tokens.append('[CLS]')
        pre_text = preprocessing_text(feature)#追加
        tokenized_text = tokenizer_mecab(pre_text)#追加
        tokens.extend(tokenized_text)#追加
        #tokens.extend(sp.encode_as_pieces(feature))# sentence piece
        tokens.append('[SEP]')
        number = len(tokens)
        numbers.append(number)

    max_token_num = max(numbers)

    return max_token_num


def get_indice(feature, maxlen, tknz):
    """
    maxlen分のID化された文を返す関数

    Parameters
    ----------------
    feature : str
        テキスト
    maxlen : int
        最大トークン数
    """

    # 最初に[CLS]、最後に'[SEP]をつけてトークン作る
    tokens = []
    pre_text = preprocessing_text(feature)
    tokens = tknz.encode(pre_text)

    #パティング（pad_sequences）後ろを埋める。長い場合は後ろを切り詰め。
    #pad_sequencesは二次元の入力が必要なため、[tokens]としてトークンの次元[0]だけ取り出す
    #indices = keras.preprocessing.sequence.pad_sequences([tokens],
    #                                                    maxlen=maxlen,
    #                                                    dtype='int32',
    #                                                    padding='post',
    #                                                    truncating='post',
    #                                                    value=0)[0]

    #パディング
    indices = np.zeros((maxlen), dtype = np.int32)
    for t, token in enumerate(tokens):
        # 最大単語数までトークンの単語をindicesに入れていく
        if t >= maxlen:
            break
        indices[t] = token

    # 最大単語数分のID化された文を返す
    return indices


def get_indice_pred(feature, maxlen, tknz):
    """
    maxlen分のID化された文を返す関数（推測用）

    Parameters
    ----------------
    feature : str
        テキスト
    maxlen : int
        最大トークン数
    """
    tokens = []
    pre_text = preprocessing_text(feature)
    tokens = tknz.encode(pre_text)

    #パティング（pad_sequences）後ろを埋める。長い場合は後ろを切り詰め。
    #pad_sequencesは二次元の入力が必要なため、[tokens]としてトークンの次元[0]だけ取り出す
    #indices = keras.preprocessing.sequence.pad_sequences([tokens],
    #                                                    maxlen=maxlen,
    #                                                    dtype='int32',
    #                                                    padding='post',
    #                                                    truncating='post',
    #                                                    value=0)[0]

    #パディング
    indices = np.zeros((maxlen), dtype=np.int32)
    for t, token in enumerate(tokens):
        if t >= maxlen:
            break
        indices[t] = token

    return indices, tokens
