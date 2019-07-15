import numpy as np

class Loss:
    """ the abstract class for all the cost functions """
    
    # calculate the cost function
    def formula(self, A, y):
        raise NotImplementedError
    
    # calculate the derivative of the cost function (dA[L]) for the last layer
    def derivative(self, A, y):
        raise NotImplementedError

class LossEntropy(Loss):
    """ Use Loss Entropy to calculate the cost """
    
    # calculate the Lose Entropy cost
    def formula(self, A, y):
        return - (y * np.log(A) + (1-y) * np.log(1-A))
    
    # calculate the derivative of the Lost Entropy cost
    def derivative(self, A, y):
        return - (np.divide(y, A)) + (np.divide(1-y, 1-A))