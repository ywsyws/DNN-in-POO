import numpy as np
from sklearn.datasets import make_moons, make_circles
from sklearn.model_selection import train_test_split

class CreateDataset:
    """ create the different datasets for the DNN model """
    
    # create a XOR dataset
    def create_xor(self, size):
        # create the entry dataset set
        self.X = np.random.randint(2, size=(2, size))
        self.y = np.sum(self.X, axis=0).reshape((1, size)) # create the lable dataset Y
        self.y[self.y != 1] = 0
        self.X = self.X + (np.random.randn(2, size) / 20) # create noises in the dataset X by adding (-0.6, 0.6) to the data
        return self.X, self.y
    
    # create a "Moon" dataset
    def create_moon(self, size):
        X, y = make_moons(n_samples=size, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.66, random_state=42)
        X_train = X_train.T
        X_test = X_test.T
        y_train = y_train.reshape(1, -1)
        y_test = y_test.reshape(1, -1)
        return X_train, X_test, y_train, y_test
    
    # create a "Circle" dataset
    def create_circle(self, size):
        X, y = make_circles(n_samples=size, noise=0.1, random_state=42)
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.66, random_state=42)
        X_train = X_train.T
        X_test = X_test.T
        y_train = y_train.reshape(1, -1)
        y_test = y_test.reshape(1, -1)
        return X_train, X_test, y_train, y_test