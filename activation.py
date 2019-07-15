import numpy as np

class Activation:
    """ the abstract class for all activation funtion classes"""
    
    # the basic formula of the activation function for the forward pass
    def formula(self, Z):
        raise NotImplementdError
    
    # to calculate the derivative of the activation function for the backward pass
    def derivative(self, input):
        raise NotImplementdError
    
    # to be used to finetune the initialized weight according to the activation function set for the first layer
    def heuristic(self, layer_b4):
        raise NotImplementdError
    

class Sigmoid(Activation):
    """ all the functions related to the sigmoid activation function """
    
    # the basic formula of the sigmoid function for the forward pass
    def formula(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    # to calculate the derivative of the sigmoid function for the backward pass
    def derivative(self, A):
        return A * (1 - A)
    
    # to be used to finetune the initialized weight if sigmoid function is set for the first layer
    def heuristic(self, layer_b4):
        return np.sqrt(1 / layer_b4)
    

class Tanh(Activation):
    """ all the functions related to the tanh activation function """
    
    # the basic formula of the tanh function for the forward pass
    def formula(self, Z):
        return (np.exp(Z) - np.exp(-Z)) / (np.exp(Z) + np.exp(-Z))
    
    # to calculate the derivative of the tanh function for the backward pass
    def derivative(self, A):
        return 1 - A**2
    
    # to be used to finetune the initialized weight if tanh function is set for the first layer
    def heuristic(self, layer_b4):
        return np.sqrt(1 / layer_b4)
    
    
class Relu(Activation):
    """ all the functions related to the relu activation function """
    
    # the basic formula of the relu function for the forward pass
    def formula(self, Z):
        return (Z > 0) * Z
    
    # to calculate the derivative of the relu function for the backward pass
    def derivative(self, Z):
        return (Z > 0) * 1
    
    # to be used to finetune the initialized weight if relu function is set for the first layer
    def heuristic(self, layer_b4):
        return np.sqrt(2 / layer_b4)