import os
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, LSTM, Dropout
from tensorflow.keras import layers, initializers, optimizers
from keras_tuner.tuners import RandomSearch
from sklearn.model_selection import train_test_split

from NN_model import readFile
from NN_model import intializeDataSet
from NN_model import reArangeDataSet


# Keras tuner
def build_model(hp):    
    model = Sequential()
    model.add(InputLayer(input_shape=i_shape,batch_size=b_size))

    model.add(LSTM(units=hp.Int('units1',min_value=1,max_value=40,step=1),activation='sigmoid',recurrent_activation='sigmoid',return_sequences=True,stateful=True,kernel_initializer=k_initializer,bias_initializer ='uniform',recurrent_initializer='Zeros',dropout=0.4,recurrent_dropout=0.1))
    # model.add(LSTM(units=hp.Int('units2',min_value=2,max_value=40,step=2),stateful=True,return_sequences=True,dropout=0.0,recurrent_dropout=0.0))
    model.add(LSTM(units=hp.Int('units2',min_value=1,max_value=40,step=1),stateful=True,dropout=0.0,recurrent_dropout=0.0))
   
    model.add(Dense(Outputs,kernel_initializer=k_initializer,bias_initializer ='uniform',activation='sigmoid'))

    model.compile(optimizer=optimizers.Adam(hp.Choice('learning_rate',values=[1e-2,1e-4]),weight_decay=0.004),loss='binary_crossentropy',metrics=['binary_accuracy'])
    return model



dirname = os.path.dirname(__file__)
filename_train = os.path.join(dirname, 'datasets/test.txt')
b_size = 200
number_of_inputs = 2
number_of_oututs = 4
time_steps = 60

X,Y = readFile(filename_train, number_of_inputs)
X_,Y_ = intializeDataSet(X,Y)
Sequential_X, Sequential_Y = reArangeDataSet(X_, Y_, time_steps)
X_train, X_val, y_train, y_val = train_test_split(Sequential_X, Sequential_Y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = X_train[len(X_train)%b_size:], X_val[len(X_val)%b_size:], y_train[len(y_train)%b_size:], y_val[len(y_val)%b_size:]

i_shape = Sequential_X[0].shape
Outputs = number_of_oututs

# weight initialize
k_initializer = initializers.GlorotNormal()
k_initializer1 = initializers.GlorotUniform()

tuner = RandomSearch(
    build_model,
    objective='val_binary_accuracy',
    max_trials=500,
    executions_per_trial=1,
    directory='HP',
    project_name='HyperParameterTunning'
)

tuner.search_space_summary()
tuner.search(X_train, y_train, validation_data=(X_val, y_val), batch_size=b_size, epochs = 10, verbose=1,shuffle=True) # Fit
tuner.results_summary()