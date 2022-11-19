import numpy as np
import re
class CommonCombinational:
    def __init__(self,logicarray,NumberOfElement):
        self.logic_array = logicarray
        self.output_pin = len(logicarray)
        self.input_pin = self.__count_input_pin(self.logic_array)
        self.NumberOfElement = NumberOfElement

    def make_truth_table(self):
        X = np.array(np.random.randint(0,high=2,size=(self.NumberOfElement, self.input_pin)),dtype='bool')
        X_ = np.vectorize(self.__bool_to_int)(X).astype(int)
        Y_ = np.zeros((self.NumberOfElement, self.output_pin),dtype='int')
        for i in range(len(Y_)):
            for j in range(len(Y_[i])):
                logic_string = self.__string_to_logic(self.logic_array[j])
                Y_[i][j] =eval(logic_string)
        return np.array(X_), np.array(Y_), self.input_pin, self.output_pin

    def __count_input_pin(self,logic_array):
        letter = []
        for logic in logic_array:
            res = [i for i in re.sub(r'[^A-Z]', '', logic)]
            letter += res
        return len(set(letter))

    def __string_to_logic(self,stringoflogic):
        list_of_cap = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        logic = ""
        for i in stringoflogic:
            if i in list_of_cap:
                logic += "X[i][{}]".format(list_of_cap.index(i))
            else:
                logic +=i
        return logic

    def __bool_to_int(self,v):
        if v == True:
            return 1
        elif v == False:
            return 0