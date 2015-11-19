#!/usr/bin/env python
import numpy as np
import tensorflow as tf

# Import MINST data
import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

Xtrain,Ytrain = mnist.nextbatch(5000)
Xtest,Ytest = mnist.nextbatch(200)

Xtrain = reshape(
