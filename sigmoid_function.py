# Sigmoid function also called a logistic function

import math
import numpy as np


# Use math.exp() for calc sigmoid for one input value
x = 3
sig = 1 / ( 1 + math.exp(-x) )
print ("\nSigmoid value =", sig)

# Use Numpy (array and exp) to cal sigmoid with multiple inputs from array
x = np.array([1, 2, 3])
sig = 1 / ( 1 + np.exp(-x) )
print ("\nSigmoid numpy array =", sig)

# Example of vector operation
print ("\nVector + 3 =", (x + 3))


# Calculating Sigmoid Gradient



