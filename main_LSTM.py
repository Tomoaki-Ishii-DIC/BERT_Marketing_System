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
#import matplotlib.pyplot as plt


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

model.summary()


# multivariate data preparation
from numpy import array
from numpy import hstack

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps):
    X, y = list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps
        print(i)
        print(end_ix)
        # check if we are beyond the dataset
        if end_ix > len(sequences):
            break
        # gather input and output parts of the pattern
        print(sequences[i:end_ix])
        print(sequences[i:end_ix, :-1])
        print(sequences[end_ix-1, -1])
        seq_x, seq_y = sequences[i:end_ix, :-1], sequences[end_ix-1, -1]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


    # choose a number of time steps
n_steps = 3
dataset = np.array(df_drop.iloc[:, 1:])

print(dataset.shape)
# convert into input/output
X, y = split_sequences(dataset, n_steps)
print(X.shape, y.shape)
print(X)

# summarize the data
for i in range(len(X)):
    print(X[i], y[i])


X_train = X[:13, :]
X_test = X[13:, :]
y_train = y[:13]
y_test = y[13:]
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


history = model.fit(X_train, y_train, batch_size = BATCHSIZE, epochs = EPOCHS)
modelName = model.__class__.__name__


model.save('./saved_model')
