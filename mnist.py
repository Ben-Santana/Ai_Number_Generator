from mll_library import layer_def as ld
from mll_library import network as net

import numpy as np

from keras.datasets import mnist
from keras.utils import to_categorical

from colorama import Fore, Back, Style


#create network
network = [
    ld.Dense(10, 30), 
    ld.Tanh(), 
    ld.Dense(30, 60), 
    ld.Tanh(),
    ld.Dense(60, 28 * 28)
]


# process MNIST data from tensorflow to train model with
def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]

# train model on MNIST number data
def trainMnist(network, numOfEpochs):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = preprocess_data(x_train, y_train, 1000)
    x_test, y_test = preprocess_data(x_test, y_test, 20)
    
    #train
    net.train(network, ld.mse, ld.mse_prime, y_train, x_train, epochs=numOfEpochs, learning_rate=0.1)