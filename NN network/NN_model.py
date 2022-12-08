import numpy as np
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import layers, initializers

number_of_input = 2
f1 = open("2_input_3bit_counter.txt", "r")
X = []
Y = []
count = 1
for line in f1:
    if count == 1:
        pass
    else:
        items = np.array([int(i) for i in line.strip().split()])
        # items[np.isclose(items, 0)] = -1
        X.append(items[:number_of_input])
        Y.append(items[number_of_input:])
    count += 1

X_ = []
Y_ = []
for i in range(len(X)):
    if i == 0:
        pass
    else:
        X_.append([i for i in X[i]]+[i for i in Y[i-1]])
        Y_.append([i for i in Y[i]])


def rearagedataset(X, Y):
    Sequential_X = []
    Sequential_Y = Y[20:]
    for i in range(len(X) - 20):
        Sequential_X.append(X[i:i + 20])
    Sequential_X = np.array(Sequential_X)
    Sequential_Y = np.array(Sequential_Y)
    return Sequential_X, Sequential_Y
Sequential_X, Sequential_Y = rearagedataset(X_, Y_ )

k_initializer=initializers.RandomUniform(minval=0.4, maxval=0.42, seed=None)

print("input_shape ",Sequential_X[0].shape)

model = Sequential()
# model.add(LSTM(64, input_shape=Sequential_X[0].shape, activation=None,return_sequences=False,stateful=True,batch_size=32,kernel_initializer=k_initializer,bias_initializer ='uniform',recurrent_initializer='Zeros'))
model.add(LSTM(64, input_shape=Sequential_X[0].shape, activation='sigmoid',return_sequences=False,kernel_initializer=k_initializer,bias_initializer ='uniform',recurrent_initializer='Zeros'))
# model.add(LSTM(32,return_sequences=False))

# model.add(LSTM(20, activation='tanh',return_sequences=True))
# model.add(layers.Flatten())
# model.add(Dense(10, activation='tanh'))

model.add(Dense(3,kernel_initializer=k_initializer,bias_initializer ='uniform',activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['binary_accuracy'])
model.fit(Sequential_X, Sequential_Y, epochs=10,verbose=2)