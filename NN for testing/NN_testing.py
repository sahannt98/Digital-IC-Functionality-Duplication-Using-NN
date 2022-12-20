# For testing
batch_size = 1
filename_test = os.path.join(dirname, 'datasets/test.txt')
X,Y = readFile(filename_test, number_of_inputs)
X_,Y_ = intializeDataSet(X,Y)
Sequential_X, Sequential_Y = reArangeDataSet(X_, Y_, batch_size, time_steps)
# new_model = newModel(Sequential_X[0], number_of_oututs, k_initializer)

# For debugging
print("\ninput_shape ",Sequential_X.shape,"\n")
print("output_shape ",Sequential_Y.shape,"\n")

new_model = Sequential()
new_model.add(LSTM(64, input_shape=Sequential_X[0],batch_size=1,activation=None,recurrent_activation='sigmoid',return_sequences=False,stateful=True,kernel_initializer=k_initializer,bias_initializer ='uniform',recurrent_initializer='Zeros'))
new_model.add(Dense(number_of_oututs,kernel_initializer=k_initializer,bias_initializer ='uniform',activation='sigmoid'))
Weights = copyWeights(model,new_model)
new_model.compile(loss='binary_crossentropy',optimizer='rmsprop', metrics=['binary_accuracy'])    


# Weights = copyWeights(model,new_model)