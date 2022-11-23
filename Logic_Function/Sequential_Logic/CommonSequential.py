import numpy as np
class CommonSequential():
    def __init__(self, path, numberofinputpin):
        self.numberofinputpin = numberofinputpin
        self.path = path
    def make_truth_table(self):
        X_ = []
        Y_ = []
        with open(self.path) as f:
            while True:
                line = f.readline()
                if not line:
                    break
                X_.append([int(i) for i in line.strip().split()[:self.numberofinputpin]])
                Y_.append([int(i) for i in line.strip().split()[self.numberofinputpin:]])
        return np.array(X_), np.array(Y_),len(X_[0]),len(Y_[0])
