import os
import wandb
from wandb.keras import WandbCallback
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import layers, initializers


from NN_model import readFile
from NN_model import intializeDataSet
from NN_model import reArangeDataSet
from NN_model import newModel
from NN_model import copyWeights
from NN_model import trainModel



dirname = os.path.dirname(__file__)
filename_test = os.path.join(dirname, 'datasets/test.txt')
batch_size = 256
number_of_inputs = 2
number_of_oututs = 3
time_steps = 40
X,Y = readFile(filename_test, number_of_inputs)
X_,Y_ = intializeDataSet(X,Y)
Sequential_X, Sequential_Y = reArangeDataSet(X_, Y_, batch_size, time_steps)

k_initializer=initializers.RandomUniform(minval=0.40, maxval=0.42, seed=None)
# new_model = tf.keras.models.load_model('NN for testing/saved_model/my_model.h5')
new_model = newModel(Sequential_X[0].shape, number_of_oututs, k_initializer, batch_size)
new_model.load_weights('NN for testing/saved_model/my_model_weights.h5')
new_model.compile(loss='binary_crossentropy', optimizer='rmsprop')
model = trainModel(new_model, Sequential_X, Sequential_Y, 5, batch_size)

