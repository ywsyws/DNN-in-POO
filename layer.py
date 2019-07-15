import numpy as np

class Layer:
    """ the abstract class for all layer classes """
    
    def __init__(self):
        pass
    
    # implement forward pass
    def forward_pass(self, input):
        raise NotImplementedError
        
    # implement backward pass
    def backward_pass(self, input):
        raise NotImplementedError

        
class FCLayer(Layer):
    
    # initialize parameters
    def __init__(self, layer_b4, layer_after, activation):

        self.activation = activation
        self.W = np.random.randn(layer_after, layer_b4) * getattr(self.activation, 'heuristic')(self, layer_b4)
        self.b = np.zeros((layer_after, 1))
    
    # calculate forward pass: linear fn (Z = WX + b) and non-linear (A = g(Z))
    def forward_pass(self, X):
        self.A_prev = X
        self.Z = np.dot(self.W, X) + self.b
        self.A = getattr(self.activation, 'formula')(self, self.Z)
        return self.A
    
    # calculate backward pass: 
    # dZ = dA * g'(Z))
    # dA[l-1] = W.T * dZ
    def backward_pass(self, dA, learning_rate):
        self.m = dA.shape[1]
        
        self.dZ = dA * getattr(self.activation, 'derivative')(self, self.A)
        pre = np.dot(self.W.T, self.dZ)
        
        self.dW = np.dot(self.dZ, self.A_prev.T) / self.m        
        
        self.W -= learning_rate * self.dW
        self.b -= learning_rate * (np.sum(self.dZ) / self.m)
        
        return np.dot(self.W.T, self.dZ) # dA[l-1]