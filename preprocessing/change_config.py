import json

def set_config(max_len):
    """
    BERTの設定ファイル内の最大トークン数を変更する関数

    Parameters
    ----------------
    max_len : int
        最大トークン
    """
    # 設定ファイルの読み込み
    json_path = "./downloads/bert-wiki-ja_config/bert_finetuning_config_v1.json"

    with open(json_path) as f:
        data = json.load(f)

    # 値の変更　最大５１２
    if max_len <= 512:
        data["max_position_embeddings"] = max_len
        data["max_seq_length"] = max_len
    else:
        data["max_position_embeddings"] = 512
        data["max_seq_length"] = 512

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

    with open(json_path) as f:
        data = json.load(f)
    print(json.dumps(data, indent=2))
