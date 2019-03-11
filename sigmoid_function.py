"""
Sigmoid function also called a logistic function
"""

import math
import numpy as np


# Use math.exp() to calculate sigmoid for ONE INPUT value
x = 3
sig = 1 / ( 1 + math.exp(-x) )
print ("\nSigmoid (one input):", sig)

# Use Numpy (array and exp) to calc sigmoid with MULTIPLE INPUTS from array
x = np.array([1, 2, 3])
sig = 1 / ( 1 + np.exp(-x) )
print ("\nSigmoid (mult inputs)", sig)

# Example of vector operation
print ("\nVector + 3 =", (x + 3))


# Calculating Sigmoid Gradient



