import pandas as pd
import numpy as np
from keras.models import load_model

import preprocessing


model = load_model('./saved_model')
model.summary()


# テキストとテーブルデータを合わせたものを用意する
# df_s.loc[11]['text']

X_f_path = "./associated_data/dataframe_all.csv"
df_X_test = pd.read_csv(X_f_path, index_col=0)#, index_col=0, header=False
df_X = df_X_test.tail(2)
print(df_X)
print(df_X_test.columns.values)

inputs_list = []
label_list = []
for df_values in df_X_test.columns.values:
    if df_values == "1":
        break
    elif df_values == "date":
        input_data = input("Please Enter \"{}\"(YYYY-MM-DD): ".format(df_values))
        inputs_list.append(input_data)
    else:
        # 入力が数値かどうかチェックする必要がある。
        input_data = int(input("Please Enter \"{}\": ".format(df_values)))
        inputs_list.append(input_data)
input_test = input("Please Enter \"Your press release text\": ")

print(inputs_list)
# テキストをID化して足す

# 5000は変更できるようにする必要あり。
text_id = preprocessing._get_indice(input_test, 5000)
print(text_id)

inputs_list.extend(text_id)

# ラベルを暫定で入力
inputs_list.append(0)
print(inputs_list)

#train_features = []
#test_features = []
#for feature in train_features_df['feature']:
#    # 上で作った関数 _get_indice  を使ってID化
#    train_features.append(preprocessing._get_indice(feature, maxlen))
## shape(len(train_features), maxlen)のゼロの行列作成
#train_segments = np.zeros((len(train_features), maxlen), dtype = np.float32)

df_inputs = pd.DataFrame([inputs_list], columns=df_X_test.columns.values)#.T

#df_inputs.columns.values = df_X_test.columns.values
print("df_X", df_X.shape)
print("df_inputs", df_inputs.shape)
#print(df_X)
#print(df_inputs)

X = pd.concat([df_X, df_inputs])
print(X)

# 日付とラベルの行を消す
X = X.iloc[:, 1:-1]
print(X)
X_test = np.array(X)

print("X_test", X_test.shape)
print(X_test)

# LSTM用の形に直す
X_test = np.asarray(X_test).astype(np.float32)
X_test = X_test[np.newaxis,:,:]
print("X_test np.newaxis", X_test.shape)


y_predict = model.predict(X_test)
print("ネガポジ出力: ", y_predict)

nega_posi = ['Positive', 'Negative']
y_pred = np.round(y_predict).astype(int)[0,0]
print("ネガポジ予測: ", nega_posi[y_pred])
