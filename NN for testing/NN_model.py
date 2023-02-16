import os
import wandb
from wandb.keras import WandbCallback
import numpy as np
import tensorflow as tf
import datetime
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras import initializers, optimizers
from keras.layers import InputLayer, Dense, LSTM, Dropout, BatchNormalization, LayerNormalization



# For the purpose of omitting "WARNING:absl:Found untraced functions"
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

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
            items = [int(x.strip().replace("'", "")) for x in line.strip().split()] 
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
def reArangeDataSet(X, Y, time_steps):
    Sequential_X = []
    Sequential_Y = Y[time_steps:]
    for i in range(len(X) - time_steps):
        Sequential_X.append(X[i:i + time_steps])
    Sequential_X = np.array(Sequential_X)
    Sequential_Y = np.array(Sequential_Y)
    return Sequential_X, Sequential_Y


# creating the NN model for training
def createModel(i_shape, b_size, Outputs, k_initializer, opt):
    model = Sequential()
    model.add(InputLayer(input_shape=i_shape,batch_size=b_size))
    model.add(LSTM(128, activation='tanh', recurrent_activation='tanh',return_sequences=True,stateful=True,kernel_initializer=k_initializer,bias_initializer ='uniform',recurrent_initializer='Zeros',dropout=0.0,recurrent_dropout=0.0))
    model.add(LSTM(128, activation='tanh', recurrent_activation='tanh',return_sequences=False,stateful=True,kernel_initializer=k_initializer,bias_initializer ='uniform',recurrent_initializer='Zeros',dropout=0.0,recurrent_dropout=0.0))
    model.add(Dense(64, activation='tanh'))
    model.add(LayerNormalization())
    model.add(Dense(Outputs,activation='sigmoid'))
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
def trainModel(model, X_train, y_train, X_val, y_val, Epochs, b_size):
    
    for i in range(Epochs):
        model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=b_size, epochs = 1, verbose=1, shuffle=True) # , callbacks=[tensorboard_callback])
        # model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=b_size, epochs = 1, verbose=1, shuffle=False, callbacks=[tensorboard_callback])
        model.reset_states()

    # model.fit(X_train, y_train, validation_data=(X_val, y_val), batch_size=b_size, epochs = Epochs, verbose=1, shuffle=False,  callbacks=[tensorboard_callback])
    return model


# copy weights & compile the model
def copyWeights(model, newModel):
    old_weights = model.get_weights()
    newModel.set_weights(old_weights)
    newModel.compile(loss='binary_crossentropy', optimizer='rmsprop')



if __name__ == "__main__":
    # Wandb
    # wandb.init(project="test-project", entity="ic-functionality-duplication",
    # config={
    # "learning_rate": 0.001,
    # "architecture": "LSTM",
    # "dataset": "4BitShiftRegisterSIPO",
    # "epochs": 10,
    # })
    
    
    # Tensorboard
    # log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    dirname = os.path.dirname(__file__)
    filename_train = os.path.join(dirname, 'datasets/4BitShiftRegisterSIPO_random2.txt')
    batch_size = 50
    number_of_inputs = 1
    number_of_oututs = 4
    time_steps = 25
    epochs = 10
    lr = 0.001

    # optimizers
    # opt = optimizers.Adam(learning_rate=lr,weight_decay=0.0005,amsgrad=F,use_ema=True,ema_momentum=0.99)
    opt = optimizers.Adam(learning_rate=lr,decay=0.004)

    # dataset preparation and then seperation for training & validation data
    X,Y = readFile(filename_train, number_of_inputs)
    X_,Y_ = intializeDataSet(X,Y)
    Sequential_X, Sequential_Y = reArangeDataSet(X_, Y_, time_steps)
    X_train, X_val, y_train, y_val = train_test_split(Sequential_X, Sequential_Y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = X_train[len(X_train)%batch_size:], X_val[len(X_val)%batch_size:], y_train[len(y_train)%batch_size:], y_val[len(y_val)%batch_size:]

    # For debugging
    print("\ninput_shape ",Sequential_X.shape,"\n")
    print("output_shape ",Sequential_Y.shape,"\n")

    # weight initialize
    k_initializer= initializers.GlorotNormal(seed=20)
    # k_initializer1=initializers.RandomUniform(minval=0.4, maxval=0.42, seed=2) 

    model = createModel(Sequential_X[0].shape, batch_size, number_of_oututs, k_initializer, opt)
    model = trainModel(model, X_train, y_train, X_val, y_val, epochs, batch_size)

    # For wights & model
    # model.save('NN for testing/saved_model/my_model.hdf5')
    # model.save_weights('NN for testing/saved_model/my_model_weights.h5')


    # For debugging
    print("\ninput_shape ",Sequential_X[0].shape,"\n")
    print("output_shape ",Sequential_Y[0].shape,"\n")
    print("inpt_3dArray_shape ",Sequential_X.shape,"\n")
    print("output_3dArray_shape ",Sequential_Y.shape,"\n")
    # print("output_shape ",len(X_),"\n")
    # print("output_shape ",Y_.shape,"\n")Sequential_Y.shape
    # print("output_shape ",Sequential_X[0],"\n")