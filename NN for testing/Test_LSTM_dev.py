import tensorflow as tf
from keras.layers import LSTM, Dense, LayerNormalization, InputLayer, BatchNormalization
from keras.models import Sequential
from keras.callbacks import Callback
from keras import initializers, optimizers
import numpy as np
from sklearn.model_selection import train_test_split
import os
dirname = os.path.dirname(__file__)
filename_train = os.path.join(dirname, 'datasets/4BitShiftRegisterSIPO_random.txt')

class ResetStatesCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        self.model.reset_states()

# Prepare the input and output data for the new circuit
timestp = 25
inputs = []
outputs = []
prev_outputs = [[0, 0, 0, 0]] * timestp # initializing previous outputs with 0
prev_inputs = [[0]] * (timestp + 1) # initializing previous inputs with 0
data = [[0, 0, 0, 0, 0]] * timestp
# Open the file
with open(filename_train, "r") as file:
    # Read the file line by line
    count = 0
    for line in file:
        # Split the line into input and output
        items = [int(x.strip().replace("'", "")) for x in line.strip().split()] 
        inp, out1, out2, out3, out4 = items[0], items[1], items[2], items[3], items[4]
        if count >= timestp+1:
            inputs.append(data)
            outputs.append([out1, out2, out3, out4])
            if count == timestp+1:
                print(inputs)
                print(outputs)
        prev_outputs = prev_outputs[1:] + [[out1, out2, out3, out4]]
        prev_inputs = prev_inputs[1:] + [[inp]]
        data = data[1:] + [[inp, prev_outputs[-2][0], prev_outputs[-2][1], prev_outputs[-2][2], prev_outputs[-2][3]]]
        count += 1

# Convert the inputs and outputs to numpy arrays
inputs = np.array(inputs, dtype=np.int64)
inputs = inputs.reshape((inputs.shape[0], timestp, 5))
inputs = inputs.astype(np.float32)
outputs = np.array(outputs)
outputs = outputs.reshape((-1, 4))
outputs = outputs.astype(np.float32)

# check the number of elements in the inputs array
print(inputs.shape)
# check the number of elements in the outputs array
print(outputs.shape)

# split the data into training and validation set
train_inputs, val_inputs, train_outputs, val_outputs = train_test_split(inputs, outputs, test_size=0.2, random_state=42)
print(train_inputs.shape, val_inputs.shape, train_outputs.shape, val_outputs.shape)

# reshaping the data to feed into LSTM
train_inputs = np.array(train_inputs).reshape((len(train_inputs), timestp, 5))
val_inputs = np.array(val_inputs).reshape((len(val_inputs), timestp, 5))
val_outputs = val_outputs.reshape((-1, 4))

batch_size = 50

# making sure that the data size is a multiple of the batch size
train_inputs = train_inputs[:batch_size*(len(train_inputs)//batch_size)]
train_outputs = train_outputs[:batch_size*(len(train_outputs)//batch_size)]
val_inputs = val_inputs[:batch_size*(len(val_inputs)//batch_size)]
val_outputs = val_outputs[:batch_size*(len(val_outputs)//batch_size)]
print(train_inputs.shape, val_inputs.shape, train_outputs.shape, val_outputs.shape)

# create the model
model = Sequential()
model.add(InputLayer(input_shape = (timestp,5),batch_size=batch_size))
model.add(LSTM(128, activation='tanh', recurrent_activation='tanh', kernel_initializer='glorot_uniform', stateful=True, return_sequences=True, bias_initializer ='uniform', recurrent_initializer='Zeros'))
model.add(LSTM(128, activation='tanh', recurrent_activation='tanh', kernel_initializer='glorot_uniform', stateful=True, return_sequences=False, bias_initializer ='uniform', recurrent_initializer='Zeros'))
model.add(Dense(64, activation='tanh'))
model.add(BatchNormalization())
model.add(Dense(4, activation='sigmoid'))
model.compile(optimizer=optimizers.Adam(learning_rate=0.001, decay=0.004), loss=tf.keras.losses.BinaryCrossentropy(), metrics=['binary_accuracy'])

# fit the model, passing in the custom callback
model.fit(train_inputs, train_outputs, batch_size=batch_size, epochs=10, validation_data=(val_inputs, val_outputs), shuffle=False, callbacks=[ResetStatesCallback()])