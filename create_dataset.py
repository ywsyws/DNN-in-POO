import numpy as np

class CreateDataset:
    """ create the 2 XOR datasets, X & Y, for the DNN model """
    
    # create the entry dataset X
    def create_X(self, X_size):
        self.X = np.random.randint(2, size=(2, X_size))
        return self.X
    
    # create the label dataset Y
    def create_Y(self, X, X_size):
        self.Y = np.sum(X, axis=0).reshape((1,X_size))
        self.Y[self.Y != 1] = 0
        return self.Y