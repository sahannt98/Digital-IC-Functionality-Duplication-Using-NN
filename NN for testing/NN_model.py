import os
import wandb
from wandb.keras import WandbCallback
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import layers, initializers

# wandb.init(project="test-project", entity="ic-functionality-duplication")

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'datasets/RingCounter_6bit.txt')


def readFile(file, number_of_input):
    f1 = open(file, "r")
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
    return X,Y


def intializeDataSet(X,Y):
    X_ = []
    Y_ = []
    for i in range(len(X)):
        if i == 0:
            pass
        else:
            X_.append([i for i in X[i]]+[i for i in Y[i-1]])
            Y_.append([i for i in Y[i]])
    return X_,Y_


def reArangeDataSet(X, Y):
    Sequential_X = []
    Sequential_Y = Y[20:]
    for i in range(len(X) - 20):
        Sequential_X.append(X[i:i + 20])
    Sequential_X = np.array(Sequential_X)
    Sequential_Y = np.array(Sequential_Y)
    return Sequential_X, Sequential_Y


# creating the NN model for training
def createModel(i_shape, b_size, Outputs, k_initializer):
    model = Sequential()
    model.add(LSTM(64, input_shape=i_shape, batch_size=b_size, activation=None,recurrent_activation='sigmoid',return_sequences=False,stateful=True,kernel_initializer=k_initializer,bias_initializer ='uniform',recurrent_initializer='Zeros'))
    # model.add(LSTM(128, input_shape=Sequential_X[0].shape, activation=None,return_sequences=True,kernel_initializer=k_initializer,bias_initializer ='uniform',recurrent_initializer='Zeros'))
    # model.add(LSTM(32,return_sequences=False))

    # model.add(LSTM(20, activation='tanh',return_sequences=True))
    # model.add(layers.Flatten())
    # model.add(Dense(10, activation='tanh'))

    model.add(Dense(Outputs,kernel_initializer=k_initializer,bias_initializer ='uniform',activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['binary_accuracy'])
    return model


# fit network / training
def trainModel(model,Sequential_X, Sequential_Y,Epochs):
    for i in range(Epochs):
        # model.fit(Sequential_X, Sequential_Y, epochs=1, verbose=2, shuffle=False, callbacks=[WandbCallback()])
        model.fit(Sequential_X, Sequential_Y, epochs = 1, verbose=2, shuffle=False)
        model.reset_states()


# copy weights
def copyWeights(model):
    old_weights = model.get_weights()
    # new_model.set_weights(old_weights)
    return old_weights


number_of_inputs = 1
X,Y = readFile(filename, number_of_inputs)
X_,Y_ = intializeDataSet(X,Y)
Sequential_X, Sequential_Y = reArangeDataSet(X_, Y_)
k_initializer=initializers.RandomUniform(minval=0.4, maxval=0.42, seed=None)
model = createModel(Sequential_X[0].shape, 32, 6, k_initializer)
trainModel(model, Sequential_X, Sequential_Y, 5)
Weights = copyWeights(model)
print(Weights)

# for debugging purposes
print("\ninput_shape ",Sequential_X[0].shape,"\n")
print("output_shape ",Sequential_Y[0].shape,"\n")
print("inpt_3dArray_shape ",Sequential_X.shape,"\n")
print("output_3dArray_shape ",Sequential_Y.shape,"\n")
# print("output_shape ",len(X_),"\n")
# print("output_shape ",Y_.shape,"\n")Sequential_Y.shape
# print("output_shape ",Sequential_X[0],"\n")
# batch_input_shape=(32,20,7)



# # summarize performance of the model
# scores = model.evaluate(Sequential_X, Sequential_Y, verbose=0)
# print("Model Accuracy: %.2f%%" % (scores[1]*100))


# # demonstrate some model predictions
# seed = [char_to_int[alphabet[0]]]
# for i in range(0, len(alphabet)-1):
#  x = np.reshape(seed, (1, len(seed), 1))
#  x = x / float(len(alphabet))
#  prediction = model.predict(x, verbose=0)
#  index = np.argmax(prediction)
#  print(int_to_char[seed[0]], "->", int_to_char[index])
#  seed = [index]
# model.reset_states()