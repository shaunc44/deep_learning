# Sigmoid function also called a logistic function

import math


x = 3
sig = 1 / ( 1 + math.exp(-x) )
print ("Sigmoid value =", sig)
