#%%
# Lets Use a MLP to clasify images of hand written digits. Later lets solve the same problem 
# but using CNN's instead of a MLP (a basic neural network)

# loading the MNIST dataset

from keras.datasets import mnist
import tensorflow as tf

(x_train, y_train),( x_test, y_test) = mnist.load_data()

print( x_train.shape )

# %%
print(y_test.shape)

# %%
