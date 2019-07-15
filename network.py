import numpy as np

class Network:
    """ build the whole L-layer DNN """
    
    def __init__(self):
        self.layers = []
    
    # combine individual layer to form the whole DNN
    def combine(self, layer):
        self.layers.append(layer)
        
    def compute_cost(self, loss, size):
        return np.sum(loss) / size
    
    # print cost during training and evaluation
    def print_cost(self,loss_fn, A, y, epoch_number=-1):
        cost = self.compute_cost(getattr(loss_fn, 'formula')(A, y), y.shape[1])
        
        # print cost during training
        if epoch_number != -1:
            print(f'cost of {epoch_number}: {cost}')
        
        # if epoch_number == -1, then we print cost during evaluation
        else:
            print(f'cost: {cost}')

    # get the derivative of the cost function for the last layer (dA[L])
    def get_error_derivative(self, loss_fn, A, y):
        return getattr(loss_fn, 'derivative')(A, y)
    
    # call forward pass function in the Layer class
    def forward(self, A):
        for layer in self.layers:
            A = layer.forward_pass(A)
        return A
    
    # call backward pass function in the Layer class
    def backward(self, dA, learning_rate):
        for layer in reversed(self.layers):
            dA = layer.backward_pass(dA, learning_rate)
        
    # train the DNN model
    def fit(self, X, y, iteration, loss_fn, learning_rate, print_freq=10000):
        for i in range(iteration):
            
            A = self.forward(X)
            if i % print_freq == 0: self.print_cost(loss_fn, A, y, epoch_number=i)        
            dA = self.get_error_derivative(loss_fn, A, y)
            self.backward(dA, learning_rate)
            
        return A
    
    # predict the result with the trained DNN model
    def predict(self, X):
        probabilities = self.forward(X)      
        predictions = (probabilities >= 0.5) * 1        
        return predictions
    
    # evaluate the performace of the DNN model
    def evaluate(self, X, y, loss_func, dataset_name="dataset", print_cost=False):
        y_hat = self.predict(X)
        accuracy = np.average((y == y_hat) * 1)
        print(f'Accuracy of the {dataset_name}: {accuracy * 100}%')
        if print_cost : self.print_cost(loss_fn, self.forward(X), y)