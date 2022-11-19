import numpy as np

class Fulladder:
    def __init__(self, NumberOfElement=10000,pinlen=5):
        self.NumberOfElement = NumberOfElement
        self.pinlen = pinlen

    def Fulladder_gate(self):
        A = np.array(np.random.randint(0,high=2,size=(self.NumberOfElement, self.pinlen)),dtype='int')
        B = np.array(np.random.randint(0,high=2,size=(self.NumberOfElement, self.pinlen)),dtype='int')
        X_ = []
        Y_ = []
        for count in range(len(A)):
            X = np.concatenate((A[count], B[count]), axis=None)
            X_.append(X)
            Y = self.__binary_sum(A[count],B[count])
            Y_.append(Y)
        X_ = np.array(X_)
        Y_ = np.array(Y_,dtype=object)
        return X_, Y_

    def __binary_sum(self,A,B):
        a=b="0b"
        for binary in range(len(A)):
            a += str(A[binary])
            b += str(B[binary])
        sum_of_ab = int(eval(a)) + int(eval(b))
        Y = np.array([int(i) for i in str(bin(sum_of_ab))[2:]])
        return Y