import os
import wandb
from wandb.keras import WandbCallback
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import layers, initializers

# Read the dataset file and seperate inputs and outputs
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


# Concatanate (n-1)th output to (n)th inputs
# New input ------> [ (n)th inputs + (n-1)th output ]
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


# Dataset reshaping/ converting 2D input array to 3D array
# 3D array ------> [Samples, Time steps, Features]
def reArangeDataSet(X, Y, batch_size, time_steps):
    Sequential_X = []
    Sequential_Y = Y[time_steps:]
    for i in range(len(X) - time_steps):
        Sequential_X.append(X[i:i + time_steps])
    Start_pt = len(Sequential_X)%batch_size
    Sequential_X = Sequential_X[Start_pt:]
    Sequential_Y = Sequential_Y[Start_pt:]
    Sequential_X = np.array(Sequential_X)
    Sequential_Y = np.array(Sequential_Y)
    return Sequential_X, Sequential_Y


# creating the NN model for training
def createModel(i_shape, b_size, Outputs, k_initializer):
    model = Sequential()
    model.add(LSTM(64, input_shape=i_shape,batch_size=b_size,activation=None,recurrent_activation='sigmoid',return_sequences=False,stateful=True,kernel_initializer=k_initializer,bias_initializer ='uniform',recurrent_initializer='Zeros'))
    # model.add(LSTM(128, input_shape=Sequential_X[0].shape, activation=None,return_sequences=True,kernel_initializer=k_initializer,bias_initializer ='uniform',recurrent_initializer='Zeros'))
    # model.add(LSTM(32,return_sequences=False))

    # model.add(LSTM(20, activation='tanh',return_sequences=True))
    # model.add(layers.Flatten())
    # model.add(Dense(10, activation='tanh'))

    model.add(Dense(Outputs,kernel_initializer=k_initializer,bias_initializer ='uniform',activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer='rmsprop', metrics=['binary_accuracy'])
    return model


# creating the NN model for testing (with batch size = 1)
def newModel(i_shape, Outputs, k_initializer):
    model = Sequential()
    model.add(LSTM(64, input_shape=i_shape, batch_size=1, activation=None,recurrent_activation='sigmoid',return_sequences=False,stateful=True,kernel_initializer=k_initializer,bias_initializer ='uniform',recurrent_initializer='Zeros'))
    model.add(Dense(Outputs,kernel_initializer=k_initializer,bias_initializer ='uniform',activation='sigmoid'))
    return model


# fit network / training
def trainModel(model, Sequential_X, Sequential_Y, Epochs, b_size):
    for i in range(Epochs):
        # model.fit(Sequential_X, Sequential_Y, epochs=1, verbose=2, shuffle=False, callbacks=[WandbCallback()])
        model.fit(Sequential_X, Sequential_Y, batch_size=b_size, epochs = 1, verbose=1, shuffle=False)
        model.reset_states()
    return model


# copy weights & compile the model
def copyWeights(model, newModel):
    old_weights = model.get_weights()
    newModel.set_weights(old_weights)
    newModel.compile(loss='binary_crossentropy', optimizer='rmsprop')


# wandb.init(project="test-project", entity="ic-functionality-duplication")

dirname = os.path.dirname(__file__)
filename_train = os.path.join(dirname, 'datasets/train.txt')
batch_size = 128
number_of_inputs = 2
number_of_oututs = 3
time_steps = 40
X,Y = readFile(filename_train, number_of_inputs)
X_,Y_ = intializeDataSet(X,Y)
Sequential_X, Sequential_Y = reArangeDataSet(X_, Y_, batch_size, time_steps)

# For debugging
print("\ninput_shape ",Sequential_X.shape,"\n")
print("output_shape ",Sequential_Y.shape,"\n")


k_initializer=initializers.RandomUniform(minval=0.40, maxval=0.42, seed=None)
model = createModel(Sequential_X[0].shape, batch_size, number_of_oututs, k_initializer)
model = trainModel(model, Sequential_X, Sequential_Y, 5, batch_size)
model.save('NN for testing/saved_model/my_model.h5')


# For debugging
print("\ninput_shape ",Sequential_X[0].shape,"\n")
print("output_shape ",Sequential_Y[0].shape,"\n")
print("inpt_3dArray_shape ",Sequential_X.shape,"\n")
print("output_3dArray_shape ",Sequential_Y.shape,"\n")
# print("output_shape ",len(X_),"\n")
# print("output_shape ",Y_.shape,"\n")Sequential_Y.shape
# print("output_shape ",Sequential_X[0],"\n")





# # online forecast
# for i in range(len(X)):
#  testX, testy = X[i], y[i]
#  testX = testX.reshape(1, 1, 1)
#  yhat = new_model.predict(testX, batch_size=n_batch)
#  print('>Expected=%.1f, Predicted=%.1f' % (testy, yhat))