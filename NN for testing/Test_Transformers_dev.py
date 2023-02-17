import numpy as np
from tensorflow import keras
from keras.layers import Input, Dense, Layer, Embedding, LayerNormalization, BatchNormalization
from keras import Model
from keras.optimizers import Adam
import keras.backend as K
import numpy as np
from sklearn.model_selection import train_test_split
import os
dirname = os.path.dirname(__file__)
filename_train = os.path.join(dirname, 'datasets/4BitShiftRegisterSIPO_random.txt')

# Prepare the input and output data for the new circuit
inputs = []
outputs = []
prev_outputs = [0, 0, 0, 0] # initializing previous outputs with 0
# Open the file
with open(filename_train, "r") as file:
    # Read the file line by line
    for line in file:
        # Split the line into input and output
        items = np.array([int(x.strip().replace("'", "")) for x in line.strip().split()])
        inp, out1, out2, out3, out4 = items[0], items[1], items[2], items[3], items[4]
        inputs.append([inp] + prev_outputs)
        outputs.append([out1, out2, out3, out4])
        prev_outputs = [out1, out2, out3, out4]


# Convert the inputs and outputs to numpy arrays
input_data  = np.array(inputs, dtype=np.int32)
output_data  = np.array(outputs, dtype=np.int32)

# split the data into training and validation set
input_train, input_test, output_train, output_test = train_test_split(input_data, output_data, test_size=0.2, random_state=42)

# Add attention layer to the deep learning network
class attention(Layer):
    def __init__(self,**kwargs):
        super(attention,self).__init__(**kwargs)
 
    def build(self,input_shape):
        self.W=self.add_weight(name='attention_weight', shape=(input_shape[-1],1), 
                               initializer='random_normal', trainable=True)
        self.b=self.add_weight(name='attention_bias', shape=(input_shape[1],1), 
                               initializer='zeros', trainable=True)        
        super(attention, self).build(input_shape)
 
    def call(self,x):
        # Alignment scores. Pass them through tanh function
        e = K.tanh(K.dot(x,self.W)+self.b)
        # Remove dimension of size 1
        e = K.squeeze(e, axis=-1)   
        # Compute the weights
        alpha = K.softmax(e)
        # Reshape to tensorFlow format
        alpha = K.expand_dims(alpha, axis=-1)
        # Compute the context vector
        context = x * alpha
        context = K.sum(context, axis=1)
        return context


def create_NN_with_attention(hidden_units, dense_units, vocabulary_size, max_length):
    inputs = Input(shape=(max_length,))
    embedding_layer = Embedding(input_dim=vocabulary_size, output_dim=hidden_units, input_length=max_length)(inputs)
    attention_layer = attention()(embedding_layer)
    dense_layer = Dense(64, activation='relu')(attention_layer)
    Normalization_layer = LayerNormalization()(dense_layer)
    outputs = Dense(dense_units, activation='sigmoid')(Normalization_layer)
    model = Model(inputs, outputs)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics=['binary_accuracy'])    
    return model       
 
# Set up parameters
hidden_units = 256
epochs = 50
batch_size = 100
dense_units = 4
vocabulary_size = 2 # since the inputs only contain 0 and 1
max_length = 5 # the number of inputs to the circuit

model_attention_embedding = create_NN_with_attention(hidden_units=hidden_units, dense_units=dense_units, vocabulary_size=vocabulary_size, max_length=max_length)

model_attention_embedding.summary() 

# Fit the model to the training data
history = model_attention_embedding.fit(input_train, output_train, epochs=epochs, batch_size=batch_size, validation_data=(input_test, output_test))