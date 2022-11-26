import numpy as np
import tensorflow as tf
from keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras import layers

class CommonNN:
    def __init__(self,X, Y, Percentage, Number_of_epochs):
        self.input_len = len(X[0])
        self.output_len = len(Y[0])
        lenth_of_train_set = int(len(X)*(Percentage/100))
        X,Y = self.__rearagedataset(X,Y)
        self.input_train = X[:lenth_of_train_set]
        self.output_train = Y[:lenth_of_train_set]
        self.input_test = X[lenth_of_train_set:]
        self.output_test = Y[lenth_of_train_set:]
        self.Number_of_epochs = Number_of_epochs


    def getAccuracy(self):
        self.__CreateModel()
        self.__TrainModel()
        Accuracy = self.__TestModel()
        return Accuracy

    def __rearagedataset(self,X,Y):
        Sequential_X = []
        Sequential_Y = Y[5:]
        for i in range(len(X)-5):
            Sequential_X.append(X[i:i+5])
        Sequential_X = np.array(Sequential_X)
        Sequential_Y = np.array(Sequential_Y)
        return Sequential_X, Sequential_Y


    def __CreateModel(self):
        self.model = Sequential()
        self.model.add(LSTM(40, input_shape=self.input_train[0].shape, activation='tanh',return_sequences=True))
        self.model.add(LSTM(20, activation='tanh',return_sequences=True))
        self.model.add(layers.Flatten())
        self.model.add(Dense(10, activation='tanh'))
        self.model.add(Dense(self.output_len, activation='tanh'))
        print(self.model.summary())
        '''
        self.model = Sequential()
        self.model.add(Dense(50, input_dim=self.input_len, activation='relu'))
        self.model.add(Dense(30, input_dim=50, activation='sigmoid'))
        self.model.add(Dense(18, input_dim=30, activation='relu'))
        self.model.add(Dense(self.output_len, activation='sigmoid'))
        '''
        self.model.compile(loss='mean_squared_error',optimizer='adam', metrics=['binary_accuracy'])

    def __TrainModel(self):
        self.model.fit(self.input_train, self.output_train, epochs=self.Number_of_epochs, verbose=2)

    def __TestModel(self):
        correct_count = 0
        for count in range(len(self.input_test)):
            predict = np.array([int(i) for i in self.model.predict(np.array([self.input_test[count]]))[0].round()])
            correct = self.output_test[count]
            if np.array_equal(correct,predict): 
                correct_count += 1
        print("Accuracy: ",np.round((correct_count*100)/len(self.input_test),3),"%") 
        return np.round((correct_count*100)/len(self.input_test),3)
        