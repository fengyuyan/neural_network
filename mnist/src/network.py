# NOTICE OF COPYRIGHT AND OWNERSHIP OF SOFTWARE
# Copyright 2017, Tom Yan. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining
# a copy of this software and associated documentation files (the
# "Software"), to deal in the Software without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to
# permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.

"""
This module implements a simple feed-forward neural network.
"""

import numpy as np
from src.activate_funs import sigmoid, sigmoid_derivative


class Network(object):
    """
    Network
    """
    def __init__(self, sizes, weights=None, biases=None, f=None, f_prime=None):
        """
        :param sizes: network sizes
        :param weights: initialized weights
        :param biases: initialized biases
        :param f: activate function, it's optional. Use Sigmod function if not specified
        :param f_prime: the derivative of activate function f
        :return:
        """
        self.layer_num = len(sizes)
        self.sizes = sizes
        if weights is not None:
            self.weights = weights
        else:
            self.weights = [np.random.randn(self.sizes[i+1], self.sizes[i]) for i in range(self.layer_num-1)]
        if biases is not None:
            self.biases = biases
        else:
            self.biases = [np.random.randn(self.sizes[i+1], 1) for i in range(self.layer_num-1)]

        self.activate_f = f if f else sigmoid
        self.activate_f_derivative = f_prime if f_prime else sigmoid_derivative

    def sgd_training(self, training_data, epochs, batch_size, eta, test_data=None):
        """
        Train network by using batch stochastic gradient descent approach
        :param training_data: a list of tuples (x, y) representing the training inputs and the desired outputs.
        :param epochs: number of epochs to train
        :param batch_size: batch size
        :param eta: learning rate
        :param test_data: test data for each epochs, optional
        :return:
        """
        pass

    def update_weight_bias(self, batch, eta):
        """
        Update network weights and biases
        :param batch: batch data used to calculate gradient
        :param eta: learning rate
        :return:
        """
        pass

    def backpropagation(self, x, y):
        """
        Calculate gradient of cost function C(w, b)
        :param x: inputs
        :param y: desired outputs
        :return: (nabla_w, nabla_b), gradient of C(w, b)
        """
        pass

    def feedforward(self, x):
        """
        Calculate network output for input x
        :param x: input
        :return: output of network
        """
        pass

    def evaluate(self, test_data):
        """
        Evaluate test data
        :param test_data:
        :return:
        """

    # 1. define network
    # 2. training: find direction and step for next iteration
    #   2.1: solve min C(x) by using back-propagation algorithm
    # 3. w1 = w0 + eta * dw
    #    b1 = b0 + eta * db
    #   terminate iteration if meet criteria
    # 4. repeat step 2 to 3
