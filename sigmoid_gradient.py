# Compute gradients to optimize loss functions using back propagation
# 1. Set s to be the sigmoid of x. You might find your sigmoid(x) function useful.
# 2. Compute σ′(x)=s(1−s)

import numpy as np


def sigmoid_derivative(x):
    """
    Compute the gradient (also called the slope or derivative) of the sigmoid function with respect to its input x.
    You can store the output of the sigmoid function into variables and then use it to calculate the gradient.
    
    Arguments:
    x -- A scalar or numpy array

    Return:
    ds -- Your computed gradient.
    """
    
    ### START CODE HERE ### (≈ 2 lines of code)
    s = 1 / ( 1 + np.exp(-x) ) # this is the SIGMOID FUNCTION
    ds = s * ( 1 - s ) # this is derivative of the SIGMOID FUNCTION
    ### END CODE HERE ###
    
    return ds


x = np.array([1, 2, 3])
print ("\nSigmoid_derivative(x) = " + str(sigmoid_derivative(x)))


