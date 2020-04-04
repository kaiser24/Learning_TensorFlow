# %%
# Loss function and optimization
# A loss function or error function is the error that our model has 
# compared to the desired output. Optimization is to take an approach to
# update our model to reduce the error (aka train our mdel)

# Lets build and train a simple linear regresion
# Y = b + W*X
# we want to find the best W and b (bias) that fits our model to predict future data

import tensorflow as tf

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
w = tf.Variable([.5], tf.float32)
b = tf.Variable([.2], tf.float32)

linear_model = b + x*w

# Lets use the sum squared error as our loss function
loss_function = tf.reduce_sum( tf.square(linear_model - y) )

# model optimizer 
optimizer = tf.train.GradientDescentOptimizer(0.01)

# Trainning node
train = optimizer.minimize(loss_function)
init = tf.global_variables_initializer()

session = tf.Session()
#variable initializer
session.run(init)
#%%
import matplotlib.pyplot as plt
# Plotting our training data
xdata = [1, 2, 3, 4, 5]
ydata = [0, -1, -2, -3, -4]

# our testing data
xtest = []
for i in range(20):
    xtest.append( 0.5*i )

plt.scatter(xdata,ydata , label='Original Data')
plt.show()

#%%

# training
iters = 6
ytest = []

# Lets plot what our model gives for our test data before training
plt.plot( xdata, session.run(linear_model, {x: xdata} ) , label = '0 iter') 
ytest.append( session.run(linear_model, {x: xtest}) )

#now lets train our data and plot for each iteration how the model behaves
for i in range(iters):
    session.run(train, {x: xdata, y: ydata})
    plt.plot( xdata, session.run(linear_model, {x: xdata} ) , label = str(i+1)+' iter')
    ytest.append( session.run(linear_model, {x: xtest}) )

plt.legend()
plt.grid(True)
plt.show()

# %%
#lets plot out model with our custom test data for each iteration of training
for i,youtput in enumerate(ytest):
    plt.plot(xtest, youtput, label = str(i)+' iter')
plt.legend()
plt.grid(True)
plt.show()

# %%
# After 3 iterations  our model gets pretty close. although, in this particular
# case we could just find the ecuation for the curve which is exact solution. but
# this was just an example. in cases with sparse data we will have to use a numerical solution.
# also this serves as an introduction for more complex models later. like neural networks and other
# machine learning algorithms

# %%
session.close()

# %%
