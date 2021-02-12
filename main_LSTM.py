import pandas as pd
import numpy as np

#from numpy import array
#from numpy import hstack
#from numpy import insert,delete

import os

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers.recurrent import LSTM
from keras.layers import Dropout
from keras.preprocessing import sequence
from keras.optimizers import RMSprop

#from keras.preprocessing.sequence import TimeseriesGenerator
#from keras.models import Sequential
#from keras.layers import Dense
from keras.layers import LSTM
from keras.callbacks import EarlyStopping
#import matplotlib.pyplot as plt


from preprocessing import make_table_text
from preprocessing import make_table_trend
from preprocessing import concat_dataset_table

# デーブルの作成（現行ではcsv出力。今後、dfの受け渡しに変更予定。）
make_table_text.make_table()
make_table_trend.make_table()
concat_dataset_table.concat()


# main_LSTMの処理

index_f_path = "./associated_data/dataframe_all.csv"
df = pd.read_csv(index_f_path, index_col=0)#, index_col=0
print(df)


BATCHSIZE = 3
EPOCHS   = 50


model = Sequential()

model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics = ['accuracy'])

#model.summary()


# multivariate data preparation
from numpy import array
from numpy import hstack

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        #print(i)
        #print(end_ix)
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        #print(sequences[i:end_ix])
        #print(sequences[i:end_ix, :-1])
        #print(sequences[end_ix-1, -1])
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


    # choose a number of time steps
n_steps = 3
dataset = np.array(df.iloc[:, 1:])

print(dataset.shape)
# convert into input/output
X, y = split_sequences(dataset, n_steps)
print(X.shape, y.shape)
#print(X)

# summarize the data
#for i in range(len(X)):
#    print(X[i], y[i])

split_rate = 0.7
split = int(X.shape[0]*split_rate)

print("Split rate: ", split_rate)
print("Train_split: ", split)
print("Val_split: ", X.shape[0]-split)

#X_train = X[:split, :]
#X_test = X[split:, :]
#y_train = y[:split]
#y_test = y[split:]
#print(X_train.shape)
#print(X_test.shape)
#print(y_train.shape)
#print(y_test.shape)


early_stopping = EarlyStopping(monitor='val_loss', mode='auto', patience=0)
history = model.fit(X, y, batch_size = BATCHSIZE, epochs = EPOCHS, validation_split=0.3, callbacks=[early_stopping])
#history = model.fit(X_train, y_train, batch_size = BATCHSIZE, epochs = EPOCHS)
#modelName = model.__class__.__name__


model.save('./saved_model')
