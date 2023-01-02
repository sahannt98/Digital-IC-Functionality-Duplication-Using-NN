import os
import wandb
from wandb.keras import WandbCallback
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, LSTM, Dropout
from tensorflow.keras import layers, initializers, optimizers


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
def createModel(i_shape, b_size, Outputs, k_initializer, opt):
    model = Sequential()
    model.add(InputLayer(input_shape=i_shape,batch_size=b_size))
    model.add(LSTM(32,activation='sigmoid',recurrent_activation='sigmoid',return_sequences=True,stateful=True,kernel_initializer=k_initializer,bias_initializer ='uniform',recurrent_initializer='Zeros',dropout=0.4,recurrent_dropout=0.1))
    model.add(LSTM(40,stateful=True,return_sequences=True,dropout=0.0,recurrent_dropout=0.0))
    model.add(LSTM(20,stateful=True,dropout=0.0,recurrent_dropout=0.0))
    model.add(Dense(Outputs,kernel_initializer=k_initializer,bias_initializer ='uniform',activation='sigmoid'))
    model.summary()
    model.compile(loss='binary_crossentropy',optimizer=opt, metrics=['binary_accuracy'])
    return model


# creating the NN model for testing (with batch size = 1)
def newModel(i_shape, Outputs, k_initializer,b_size=1):
    model = Sequential()
    model.add(InputLayer(input_shape=i_shape, batch_size=b_size))
    model.add(LSTM(64, activation=None,recurrent_activation='sigmoid',return_sequences=False,stateful=True,kernel_initializer=k_initializer,bias_initializer ='uniform',recurrent_initializer='Zeros'))
    model.add(Dense(Outputs,kernel_initializer=k_initializer,bias_initializer ='uniform',activation='sigmoid'))
    return model


# fit network / training
def trainModel(model, Sequential_X, Sequential_Y, Epochs, b_size):
    for i in range(Epochs):
        # model.fit(Sequential_X, Sequential_Y, batch_size=b_size, epochs = 1, verbose=1, shuffle=False, callbacks=[WandbCallback()])
        model.fit(Sequential_X, Sequential_Y, batch_size=b_size, epochs = 1, verbose=1, shuffle=False)
        model.reset_states()
    return model


# copy weights & compile the model
def copyWeights(model, newModel):
    old_weights = model.get_weights()
    newModel.set_weights(old_weights)
    newModel.compile(loss='binary_crossentropy', optimizer='rmsprop')

if __name__ == "__main__":
    # wandb.init(project="test-project", entity="ic-functionality-duplication")

    dirname = os.path.dirname(__file__)
    filename_train = os.path.join(dirname, 'datasets/9BitCounter.txt')
    batch_size = 10
    number_of_inputs = 1
    number_of_oututs = 9
    time_steps = 60
    epochs = 1000
    lr = 0.0001

    # optimizers
    opt = optimizers.Adam(learning_rate=lr,weight_decay=0.004)
    opt1 = optimizers.experimental.AdamW(learning_rate=lr,weight_decay=0.004)
    opt2 = optimizers.SGD(learning_rate=lr,weight_decay=0.004,momentum=0.0)
    opt3 = optimizers.RMSprop(learning_rate=lr,weight_decay=0.004,momentum=0.0)
    opt4 = optimizers.Nadam(learning_rate=lr,weight_decay=0.004)

    X,Y = readFile(filename_train, number_of_inputs)
    X_,Y_ = intializeDataSet(X,Y)
    Sequential_X, Sequential_Y = reArangeDataSet(X_, Y_, batch_size, time_steps)

    # For debugging
    print("\ninput_shape ",Sequential_X.shape,"\n")
    print("output_shape ",Sequential_Y.shape,"\n")

    # weight initialize
    k_initializer= initializers.GlorotNormal()
    k_initializer1=initializers.RandomUniform(minval=0.4, maxval=0.42, seed=2) 

    model = createModel(Sequential_X[0].shape, batch_size, number_of_oututs, k_initializer, opt)
    model = trainModel(model, Sequential_X, Sequential_Y, epochs, batch_size)
    model.save('NN for testing/saved_model/my_model.h5')
    model.save_weights('NN for testing/saved_model/my_model_weights.h5')


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