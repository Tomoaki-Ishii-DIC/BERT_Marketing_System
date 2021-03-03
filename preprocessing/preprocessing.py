import MeCab
import re
import string
import sentencepiece as spm
import numpy as np

def preprocessing_text(text):
    '''
    前処理
    '''
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

wakati = MeCab.Tagger("-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")

def tokenizer_mecab(text):
    '''
    分かち書き
    '''
    #wakati = MeCab.Tagger("-Owakati -d /usr/local/lib/mecab/dic/mecab-ipadic-neologd")
    words = wakati.parse(text).split()

    return words


def get_max(X):
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
        #print("テキストのデータ :\n",tokens)
        number = len(tokens)
        numbers.append(number)

    max_token_num = max(numbers)
    #print("Maximum number of words: " + str(max_token_num))

    return max_token_num



# ここでもsentence piece
sp = spm.SentencePieceProcessor()
sp.Load('./downloads/bert-wiki-ja/wiki-ja.model')

# 最大単語数分のID化された文を返す関数
# maxlenがなくてエラーになるので勝手に追加（maxlenは最大単語数か？）
def get_indice(feature, maxlen):
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

def get_indice_pred(feature, maxlen):
    indices = np.zeros((maxlen), dtype=np.int32)

    tokens = []
    tokens.append('[CLS]')
    pre_text = preprocessing_text(feature)#追加
    tokenized_text = tokenizer_mecab(pre_text)#追加
    tokens.extend(tokenized_text)#追加
    #tokens.extend(sp.encode_as_pieces(feature))
    tokens.append('[SEP]')

    for t, token in enumerate(tokens):
        if t >= maxlen:
            break
        try:
            indices[t] = sp.piece_to_id(token)
        except:
            logging.warn('unknown')
            indices[t] = sp.piece_to_id('<unk>')

    return indices, tokens
