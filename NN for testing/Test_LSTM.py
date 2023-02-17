import tensorflow as tf
from keras.layers import LSTM, Dense, LayerNormalization
from keras.models import Sequential
from keras.callbacks import Callback
from keras import initializers, optimizers
import numpy as np
from sklearn.model_selection import train_test_split
import os
dirname = os.path.dirname(__file__)
filename_train = os.path.join(dirname, 'datasets/4BitShiftRegisterSIPO_random.txt')

batch_size = 100
features = 5
inps = 1
outps = 4


class ResetStatesCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.reset_states()
        print(" -> Resetting model states at end of epoch ", epoch)

# Prepare the input and output data for the new circuit
inputs = []
outputs = []
prev_outputs = [0]*outps # initializing previous outputs with 0
# Open the file
with open(filename_train, "r") as file:
    # Read the file line by line
    for line in file:
        # Split the line into individual list of integers
        items = [int(x.strip().replace("'", "")) for x in line.strip().split()] 
        inp1, out1, out2, out3, out4 = items[0], items[1], items[2], items[3], items[4]
        inputs.append([inp1] + prev_outputs)
        outputs.append([out1, out2, out3, out4])
        for i in range(1, outps + 1):
            output = "out{}".format(i)
            prev_outputs.append(output)
        prev_outputs = [out1, out2, out3, out4] 


# Convert the inputs and outputs to numpy arrays
inputs = np.array(inputs, dtype=np.int64)
inputs = inputs.reshape((inputs.shape[0], 1, features))
inputs = inputs.astype(np.float32)
outputs = np.array(outputs)
outputs = outputs.reshape((-1, outps))

print(inputs)

# check the number of elements in the inputs array
print(inputs.shape[0])
# check the number of elements in the outputs array
print(outputs.shape[0])

# split the data into training and validation set
train_inputs, val_inputs, train_outputs, val_outputs =  train_test_split(inputs, outputs, test_size=0.2, random_state=42)
print(train_inputs.shape, val_inputs.shape, train_outputs.shape, val_outputs.shape)

# reshaping the data to feed into LSTM
train_inputs = np.array(train_inputs).reshape((len(train_inputs), 1, features))
train_outputs = train_outputs.reshape((-1, outps))
val_inputs = np.array(val_inputs).reshape((len(val_inputs), 1, features))
val_outputs = val_outputs.reshape((-1, outps))

# making sure that the data size is a multiple of the batch size
train_inputs = train_inputs[:batch_size*(len(train_inputs)//batch_size)]
train_outputs = train_outputs[:batch_size*(len(train_outputs)//batch_size)]
val_inputs = val_inputs[:batch_size*(len(val_inputs)//batch_size)]
val_outputs = val_outputs[:batch_size*(len(val_outputs)//batch_size)]
print(train_inputs.shape, val_inputs.shape, train_outputs.shape, val_outputs.shape)

# create the model
model = Sequential()
model.add(LSTM(128,activation='tanh',recurrent_activation='tanh', kernel_initializer='glorot_normal', stateful=True, return_sequences=True, batch_input_shape=(batch_size, 1, features), bias_initializer ='uniform', recurrent_initializer='Zeros'))
model.add(LSTM(128,activation='tanh',recurrent_activation='tanh', kernel_initializer='glorot_normal', stateful=True, return_sequences=False, bias_initializer ='uniform', recurrent_initializer='Zeros'))
model.add(Dense(32, activation='tanh'))
model.add(LayerNormalization())
model.add(Dense(outps, activation='sigmoid'))
model.compile(optimizer=optimizers.Adam(learning_rate=0.001,decay=0.004), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['binary_accuracy'])

# fit the model, passing in the custom callback
model.fit(train_inputs, train_outputs, batch_size=batch_size, epochs=10, validation_data=(val_inputs, val_outputs), shuffle=False, callbacks=[ResetStatesCallback()])