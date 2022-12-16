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
batch_size = 64

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
    Sequential_Y = Y[40:]
    for i in range(len(X) - 40):
        Sequential_X.append(X[i:i + 40])
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
    model.compile(loss='binary_crossentropy',optimizer='adam', metrics=['binary_accuracy'])
    return model


# creating the NN model for testing (with batch size = 1)
def newModel(i_shape, Outputs, k_initializer):
    model = Sequential()
    model.add(LSTM(10, input_shape=i_shape, batch_size=1, activation=None,recurrent_activation='sigmoid',return_sequences=False,stateful=True,kernel_initializer=k_initializer,bias_initializer ='uniform',recurrent_initializer='Zeros'))
    model.add(Dense(Outputs,kernel_initializer=k_initializer,bias_initializer ='uniform',activation='sigmoid'))
    return model


# fit network / training
def trainModel(model, Sequential_X, Sequential_Y, Epochs, b_size):
    for i in range(Epochs):
        # model.fit(Sequential_X, Sequential_Y, epochs=1, verbose=2, shuffle=False, callbacks=[WandbCallback()])
        model.fit(Sequential_X, Sequential_Y, batch_size=b_size, epochs = 1, verbose=2, shuffle=False)
        model.reset_states()


# copy weights & compile the model
def copyWeights(model, newModel):
    old_weights = model.get_weights()
    newModel.set_weights(old_weights)
    newModel.compile(loss='binary_crossentropy', optimizer='adam')


number_of_inputs = 1
X,Y = readFile(filename, number_of_inputs)
X_,Y_ = intializeDataSet(X,Y)
Sequential_X, Sequential_Y = reArangeDataSet(X_, Y_)
print("\ninput_shape ",Sequential_X.shape,"\n")
print("output_shape ",Sequential_Y.shape,"\n")




k_initializer=initializers.RandomUniform(minval=0.395, maxval=0.415, seed=None)
model = createModel(Sequential_X[0].shape, batch_size, 6, k_initializer)
trainModel(model, Sequential_X, Sequential_Y, 1000, batch_size)
# new_model = newModel(i_shape, Outputs, k_initializer)
# Weights = copyWeights(model,new_model)



print("\ninput_shape ",Sequential_X[0].shape,"\n")
print("output_shape ",Sequential_Y[0].shape,"\n")
print("inpt_3dArray_shape ",Sequential_X.shape,"\n")
print("output_3dArray_shape ",Sequential_Y.shape,"\n")
# print("output_shape ",len(X_),"\n")
# print("output_shape ",Y_.shape,"\n")Sequential_Y.shape
# print("output_shape ",Sequential_X[0],"\n")
# batch_input_shape=(32,20,7)



# # online forecast
# for i in range(len(X)):
#  testX, testy = X[i], y[i]
#  testX = testX.reshape(1, 1, 1)
#  yhat = new_model.predict(testX, batch_size=n_batch)
#  print('>Expected=%.1f, Predicted=%.1f' % (testy, yhat))