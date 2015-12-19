#!/usr/bin/env python
import numpy as np
import tensorflow as tf

# Import MINST data
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

Xtrain,Ytrain = mnist.train.next_batch(5000)
Xtest,Ytest = mnist.test.next_batch(200)

print Xtrain.shape
print Ytrain.shape
print Xtest.shape
print Ytest.shape
#print shape(Xtrain)

Xtrain = np.reshape(Xtrain, newshape=(-1, 28*28))
Xtest = np.reshape(Xtest, newshape=(-1, 28*28))

print Xtrain.shape
print Xtest.shape
