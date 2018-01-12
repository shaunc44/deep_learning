import numpy as numpy


# creates five random Gaussian variables stored in array a
a = np.random.randn(5)
print (a)

# this is a rank 1 array
# don't use shapes like this
print (a.shape)

# print a transposed
print (a.T)

print (np.dot(a,a.T))

a = np.random.randn(5, 1) # a.shape = (5, 1) "column vector"
print (a)

a = np.random.randn(1, 5) # a.shape = (1, 5) "row vector"
print (a)

print (a.T)

print (np.dot(a, a.T))


# test whether vector is correct shape or not
assert( a.shape == (5, 1) )


# to reshape rank 1 array to column or row vector
a = a.reshape((5, 1))







