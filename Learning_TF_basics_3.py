#%%
# Lets Use a MLP to clasify images of hand written digits. Later lets solve the same problem 
# but using CNN's instead of a MLP (a basic neural network)

# loading the MNIST dataset

from keras.datasets import mnist
import tensorflow as tf

(x_train, y_train),( x_test, y_test) = mnist.load_data()

# %%
# Our data consists of images with 28,28 pixels. N,28,28. N images.
# our input layer will be the 28*28 pixels of the image so we have to reshape our data before
# putting it into our model.

# Before that we need to normilize our data. for a digital grayscale image each pixel has a value
# from 0-255 and we need to normalize it.
x_train = (x_train.astype(float) / 255 )
x_test = (x_train.astype(float) / 255 )  

# now reshape from N,28,28 -> N,28*28
x_train = x_train.reshape( ( x_train.shape[0], x_train.shape[1]*x_train.shape[2] )  )
x_test = x_test.reshape( ( x_test.shape[0], x_test.shape[1]*x_test.shape[2] )  )

print(x_train.shape, x_test.shape)