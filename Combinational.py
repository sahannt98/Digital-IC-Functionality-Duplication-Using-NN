import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense

class Combinational:
    def __init__(self,X, Y, input_len, output_len):
        self.input_len = input_len
        self.output_len = output_len
        lenth_of_train_set = int(len(X)*0.9)
        self.input_train = X[:lenth_of_train_set]
        self.output_train = Y[:lenth_of_train_set]
        self.input_test = X[lenth_of_train_set:]
        self.output_test = Y[lenth_of_train_set:]
        self.__CreateModel()
        self.__TrainModel()
        self.__TestModel()

    def __CreateModel(self):
        self.model = Sequential()
        self.model.add(Dense(50, input_dim=self.input_len, activation='relu'))
        self.model.add(Dense(30, input_dim=50, activation='sigmoid'))
        self.model.add(Dense(18, input_dim=30, activation='relu'))
        self.model.add(Dense(self.output_len, activation='sigmoid'))
        self.model.compile(loss='mean_squared_error',optimizer='adam', metrics=['binary_accuracy'])

    def __TrainModel(self):
        self.model.fit(self.input_train, self.output_train, epochs=30, verbose=2)

    def __TestModel(self):
        currect_count = 0
        for count in range(len(self.input_test)):
            predict = np.array([int(i) for i in self.model.predict(np.array([self.input_test[count]]))[0].round()])
            currect = self.output_test[count]
            if np.array_equal(currect,predict): 
                currect_count += 1 
        print("Acurasy: ",np.round((currect_count*100)/len(self.input_test),2),"%")