from create_dataset import CreateDataset
from layer import FCLayer
from loss import LossEntropy
from network import Network
from activation import Relu, Sigmoid
import matplotlib.pyplot as plt

#
# DATASET CREATION
#
size = 4000
ds = CreateDataset()
# create Circle dataset
ds.create_circle(size)
X_train, X_test, y_train, y_test = ds.create_circle(size)


#
# NEURAL NETWORK
#

learning_rate = 0.1
loss_fn = LossEntropy()
iteration = 100000

net = Network()
net.combine(FCLayer(2, 4, Relu))
net.combine(FCLayer(4, 6, Relu))
net.combine(FCLayer(6, 3, Relu))
net.combine(FCLayer(3, 1, Sigmoid))

# train the DNN model
A = net.fit(X_train, y_train, iteration, loss_fn, learning_rate)

# predict a result with a test dataset using the trained DNN model
net.evaluate(X_test, y_test, loss_fn, dataset_name="test dataset", print_cost=True)


#
# DATA VISUALISATION 
#
plt.scatter(X_train[0,:], X_train[1,:], c=y_train[0,:])
y_pred = net.predict(X_test)
plt.scatter(X_test[0,:], X_test[1,:], c=y_pred[0,:])