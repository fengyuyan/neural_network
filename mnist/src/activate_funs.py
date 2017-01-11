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
This module implements several common activate functions.
"""
import numpy as np


def sigmoid(x):
    """
    Sigmoid function
    :param x:
    :return:
    """
    return 1.0/(1.0 + np.exp(-x))


def sigmoid_derivative(x):
    """
    Derivative of sigmoid function
    :param x:
    :return:
    """
    return sigmoid(x)*(1-sigmoid(x))

