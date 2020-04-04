# %%

# Learning Tensorflow 1.
# In Tensorflow we must create (like defs on other languages) the constants or variables that we will use later
# the points where these values interact with each other are nodes. These nodes can be functions etc.
# operations and variables are described as nodes on a graph and then run as a tf session

import tensorflow as tf
# %%
# simple sum

# two input nodes that are constant values
node1 = tf.constant(5.0)
node2 = tf.constant(4.0)
# a sum node
tfsum = tf.add(node1, node2)

# opens the session
session = tf.Session()
# runs and prints the graph
print( session.run(tfsum) )
session.close()


# %%
# sum of variables

# we can leave variables currently uknown for future usage as placeholders
val1 = tf.placeholder(tf.float32)
val2 = tf.placeholder(tf.float32)
tfsum = tf.add(val1, val2)

session2 = tf.Session()
# We run the graph and input the values as a dictionary. 
print( session2.run( tfsum, feed_dict={val1:[2,3,4], val2:[5,2,1]} ) )
session2.close()


#%%
# Lets do some matrix operations

# input matrices
mat1 = tf.placeholder(tf.float32)
mat2 = tf.placeholder(tf.float32)
# Dot product. aka matrix multiplication
mulnode = tf.linalg.matmul(mat1, mat2)
# matrix transpose

session3 = tf.Session()
newmat = session3.run( mulnode, feed_dict={ mat1:[[2,2,3],
[5,4,3],
[2,3,4]], mat2:[[1,1,0],
[1,0,0],
[1,2,1]] } )

print(newmat)
session3.close()

# %%

session4 = tf.Session()
newmat2 = session4.run( mulnode, feed_dict={ mat1:[[2,2,3]], mat2:[[1],[1],[5]]} )

print(newmat2)
session4.close()

# %%
# Variables. data that we assign a value initially but which can change
# later in our code. basicly for trainning etc.

W1 = tf.Variable( [.5], tf.float32)
# In tensorflow we have to initialize the variables, and this action is also a node in our graph.
init = tf.global_variables_initializer()

session5 = tf.Session()
just_initializer = session5.run( init )
print( session5.run(W1) )
session5.close()
# %%

# Activation Functions in TF
# Activation Functions are functions used on nodes of a graph (like on a neuron on a ANN)
# takes an input and depending on the type of activation function it triggers an output.

# Lets visualize some Activation functions
import matplotlib.pyplot as plt
import numpy as np
x = np.linspace(-5, 5, 100)
sigmoid = tf.sigmoid(x)
tanh = tf.tanh(x)
relu = tf.nn.relu(x)

with tf.Session() as session6:
    sigmoid_out = session6.run( sigmoid )
    tanh_out = session6.run( tanh )
    relu_out = session6.run( relu )

plt.plot(x, sigmoid_out, label='Sigmoid')
plt.plot(x,tanh_out, label='tanh')
plt.plot(x, relu_out, label='reLU')
plt.grid(True)
plt.legend()
plt.show()

# %%
# One Hot Encodding
# One hot encoding is to take an integer number which represents the id of a class label, and encode it as
# a vector of zeros and a one for the id it represents.
# example. if we have 3 classes.
# 1,2,3.
# the first one would be [ 1 0 0], the second [0 1 0 ] and the third [ 0 0 1]
# if we have a vector with labels like this
Y = [1 ,2 ,1, 3 , 1, 3, 3]
# The encoding would be
Yen = [ [1,0,0],[0,1,0],[1,0,0],[0,0,1],[1,0,0],[0,0,1],[0,0,1] ]

#in Tf we can use the function tf.one_hot

# %%
