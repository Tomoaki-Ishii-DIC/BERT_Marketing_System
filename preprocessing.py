import MeCab
import re
import string

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
    print("Maximum number of words: " + str(max_token_num))

    return max_token_num
