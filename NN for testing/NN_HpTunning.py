import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, LSTM, Dropout
from tensorflow.keras import layers, initializers, optimizers
from keras_tuner.tuners import RandomSearch

from NN_model import readFile
from NN_model import intializeDataSet
from NN_model import reArangeDataSet
from NN_model import createModel
from NN_model import trainModel


# Keras tuner
def build_model(hp):
    model = Sequential()
    model.add(InputLayer(input_shape=i_shape,batch_size=b_size))

    model.add(LSTM(units=hp.Int('units1',min_value=2,max_value=40,step=2),activation=None,recurrent_activation='sigmoid',return_sequences=True,stateful=True,kernel_initializer=k_initializer,bias_initializer ='uniform',recurrent_initializer='Zeros',dropout=0.4,recurrent_dropout=0.1))
    model.add(LSTM(units=hp.Int('units2',min_value=2,max_value=40,step=2),stateful=True,return_sequences=True,dropout=0.0,recurrent_dropout=0.0))
    model.add(LSTM(units=hp.Int('units3',min_value=2,max_value=40,step=2),stateful=True,dropout=0.0,recurrent_dropout=0.0))
    model.add(Dense(Outputs,kernel_initializer=k_initializer,bias_initializer ='uniform',activation='sigmoid'))

    model.compile(optimizer=optimizers.Adam(hp.Choice('learning_rate',values=[1e-2,1e-5])),loss='binary_crossentropy',metrics=['binary_accuracy'])
    return model



dirname = os.path.dirname(__file__)
filename_train = os.path.join(dirname, 'datasets/9BitCounter.txt')
b_size = 60
number_of_inputs = 1
number_of_oututs = 9
time_steps = 60

X,Y = readFile(filename_train, number_of_inputs)
X_,Y_ = intializeDataSet(X,Y)
Sequential_X, Sequential_Y = reArangeDataSet(X_, Y_, b_size, time_steps)
k_initializer=initializers.RandomUniform(minval=0.4, maxval=0.42, seed=None) # weight initialize

i_shape=Sequential_X[0].shape
Outputs=number_of_oututs

tuner = RandomSearch(
    build_model,
    objective='binary_accuracy',
    max_trials=4,
    executions_per_trial=1,
    directory='HP',
    project_name='HyperParameterTunning'
)

tuner.search_space_summary()
tuner.search(Sequential_X,Sequential_Y,batch_size=b_size,epochs=1000,shuffle=False) # Fit
tuner.results_summary()
