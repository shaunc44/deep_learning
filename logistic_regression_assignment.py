import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

%matplotlib inline


"""
Problem Statement: You are given a dataset ("data.h5") containing:

- a training set of m_train images labeled as cat (y=1) or non-cat (y=0)
- a test set of m_test images labeled as cat or non-cat
- each image is of shape (num_px, num_px, 3) where 3 is for the 3 channels (RGB). 
- Thus, each image is square (height = num_px) and (width = num_px).

You will build a simple image-recognition algorithm 
that can correctly classify pictures as cat or non-cat.
"""


# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

"""
We added "_orig" at the end of image datasets (train and test) because we are going to preprocess them. 
After preprocessing, we will end up with train_set_x and test_set_x 
(the labels train_set_y and test_set_y don't need any preprocessing).

Each line of your train_set_x_orig and test_set_x_orig is an array representing an image. 
You can visualize an example by running the following code. 
Feel free also to change the index value and re-run to see other images.
"""


# Example of a picture
index = 25
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) 
        + ", it's a '" 
        + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") 
        + "' picture.")


"""
Many software bugs in deep learning come from having matrix/vector dimensions that don't fit. 
If you can keep your matrix/vector dimensions straight you will go a long way toward eliminating many bugs.

Exercise: Find the values for:

- m_train (number of training examples)
- m_test (number of test examples)
- num_px (= height = width of a training image)

Remember that train_set_x_orig is a numpy-array of shape (m_train, num_px, num_px, 3). 
For instance, you can access m_train by writing train_set_x_orig.shape[0].
"""


### START CODE HERE ### (≈ 3 lines of code)
m_train = None
m_test = None
num_px = None
### END CODE HERE ###

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))














