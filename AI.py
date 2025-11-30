import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
import time

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# Reshape x_train and x_test to flatten the 28x28 images into a 784-element vector
x_train = x_train.reshape(x_train.shape[0], 784)
x_test = x_test.reshape(x_test.shape[0], 784)

assert x_train.shape == (60000, 784)
assert x_test.shape == (10000, 784)
assert y_train.shape == (60000,)
assert y_test.shape == (10000,)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

def sig(x):
    #Defining the sigmoid function
    return 1/(1+np.exp(-x))
def sigd(x):
    s = sig(x)
    return s * (1 - s)
def random(x, y):
    return 2*(np.random.rand(x, y))-1

def softm(x):
    e_x = np.exp(x - np.max(x))  # Subtract max for numerical stability
    return e_x / np.sum(e_x)

input_size = 784
data_size = 60000

hl_total = int(input_size*(2/3))       #Total number of nodes in hidden layer 1
hl2_total = int(hl_total*(2/3))        #Total number of nodes in hidden layer 2
output_number = 10

weights_array1 = random(hl_total, input_size)                     #The weights for the set of numbers to hidden layer 1
bias_array1 = np.random.rand(hl_total, 1)                          #Biases for the first hidden layer connections

weights_array2 = random(hl2_total, hl_total)               #Weights for next hidden layer
bias_array2 = np.random.rand(hl2_total, 1)                         #Biases for hidden layer 2

weights_array3 = random(output_number, hl2_total)          #Weights for output
bias_array3 = np.random.rand(output_number, 1)                                     #Biases for output
l=3
#Training#

for i in range(data_size):
  #Initialize#
    target_vector = np.zeros((output_number, 1))
    target_vector.reshape(-1,1)
  #FORWARD-PROPOGATION#

    array_input = x_train[i].reshape(-1,1)              #Input array that the nueral network takes in

    sum1 = (np.dot(weights_array1, array_input))+(bias_array1)
    hl_val = sig(sum1)  #Values for hidden layer 1

    sum2 = (np.dot(weights_array2, hl_val)+(bias_array2))
    hl2_val = sig(sum2)      #Values for hidden layer 2

    sum3 = (np.dot(weights_array3, hl2_val))+(bias_array3)
    output_val = softm(sum3)  #output values

  #BACK-PROPOGATION#
    print(output_val)
    target_vector[y_train[i]]=1
    Cost = -1*(np.log(output_val[y_train[i], 0])+1e-12)
    zL = [0, sum1, sum2, sum3]
    wL = [0, weights_array1, weights_array2, weights_array3]
    while l>0:
        for n in range(hl2_total):
            for n2 in range(output_number):
                weight=((wL[l])[n2])[n]
                activation

        l=l-1

    time.sleep(30)