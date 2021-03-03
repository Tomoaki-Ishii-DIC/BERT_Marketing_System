import json

def set_config(max_len):
    # ローカルJSONファイルの読み込み
    json_path = "./downloads/bert-wiki-ja_config/bert_finetuning_config_v1.json"

    with open(json_path) as f:
        data = json.load(f)

    # 値の変更
    data["max_position_embeddings"] = max_len
    data["max_seq_length"] = max_len

    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)

    with open(json_path) as f:
        data = json.load(f)
    print(json.dumps(data, indent=2))
