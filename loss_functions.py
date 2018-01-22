# The loss is used to evaluate the performance of your model. 
# The bigger your loss is, the more different 
# your predictions (ŷ) are from the true values (y). 
# In deep learning, you use optimization algorithms like Gradient Descent 
# to train your model and to minimize the cost.

import numpy as np


def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L1 loss function defined above
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    loss = np.sum( abs(y - yhat) )
    ### END CODE HERE ###
    
    return loss


yhat = np.array([0.9, 0.2, 0.1, 0.4, 0.9])
y = np.array([1, 0, 0, 1, 1])
print("\nL1 = " + str(L1(yhat,y)))


# There are several way of implementing the L2 loss, 
# but you may find the function np.dot() useful. 
# As a reminder, if x=[x1,x2,...,xn], then np.dot(x,x) .... 
def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)
    
    Returns:
    loss -- the value of the L2 loss function defined above
    """
    
    ### START CODE HERE ### (≈ 1 line of code)
    z = y - yhat
    loss = np.dot(z, z)
    ### END CODE HERE ###
    
    return loss


yhat = np.array([0.9, 0.2, 0.1, 0.4, 0.9])
y = np.array([1, 0, 0, 1, 1])
print("\nL2 = " + str(L2(yhat,y)))





