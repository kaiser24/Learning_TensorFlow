#%%
# Lets Use a MLP to clasify images of hand written digits. Later lets solve the same problem 
# but using CNN's instead of a MLP (a basic neural network)

# loading the MNIST dataset

from keras.datasets import mnist
import tensorflow as tf
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
import numpy as np

(x_train, y_train),( x_test, y_test) = mnist.load_data()
lb = LabelBinarizer()



# %%
# Our data consists of images with 28,28 pixels. N,28,28. N images.
# our input layer will be the 28*28 pixels of the image so we have to reshape our data before
# putting it into our model.

# Before that we need to normilize our data. for a digital grayscale image each pixel has a value
# from 0-255 and we need to normalize it.
x_train = (x_train.astype(float) / 255 )
x_test = (x_test.astype(float) / 255 )  

# now reshape from N,28,28 -> N,28*28
x_train = x_train.reshape( ( x_train.shape[0], x_train.shape[1]*x_train.shape[2] )  )
x_test = x_test.reshape( ( x_test.shape[0], x_test.shape[1]*x_test.shape[2] )  )
print(np.min(y_train) ,np.max(y_train))
y_train = lb.fit_transform(y_train)
print(y_train.argmax(1)[2])
print("Data sizes")
print("x train: {} y train: {}".format(x_train.shape, y_train.shape))
print("x test: {} y test: {}".format(x_test.shape, y_test.shape))

session = tf.Session()

# Our Model
# 784 -> 512 -> 256 -> 10 . output are 10 classes

numInputFeatures = x_train.shape[1]
numNeuronsLayer1 = 512
numNeuronsLayer2 = 256
numOutputClasses = y_train.shape[1]
starterLearningRate = 0.001
regularizerRate = 0.1

#===input data===
inputX = tf.placeholder('float32', shape=(None,numInputFeatures), name='inputX')
inputY = tf.placeholder('float32', shape=(None,numOutputClasses), name='inputX')
keepProb = tf.placeholder(tf.float32)


#===Variables (weights and biases) ===

# For the weights we have a matrix of size N(i-1),Ni.
# where i indicates the next layer and i-1 the previous layer.
# Biases are just values to sum to the neurons of the i-th layer
# we use tf.random_normal() to initialize the variables at random values with a given std dev

weights01 = tf.Variable( tf.random_normal( [numInputFeatures,numNeuronsLayer1] ,stddev=(1/tf.sqrt(float(numInputFeatures)) ) ) )
bias1 = tf.Variable( tf.random_normal( [numNeuronsLayer1] ) )

weights12 = tf.Variable( tf.random_normal( [numNeuronsLayer1,numNeuronsLayer2] ,stddev=(1/tf.sqrt(float(numNeuronsLayer1)) ) ) )
bias2 = tf.Variable( tf.random_normal( [numNeuronsLayer2] ) )

weights23 = tf.Variable( tf.random_normal( [numNeuronsLayer2,numOutputClasses] ,stddev=(1/tf.sqrt(float(numNeuronsLayer2)) ) ) )
bias3 = tf.Variable( tf.random_normal( [numOutputClasses] ) )

# the graph itself. We apply a dot product between the i-1 layer 
# and the weight and then we add biases. L(i-1) * W(i-1)i + Bi = Li
# we also apply dropout after each layer
partOutputLayer1 = tf.nn.relu( tf.matmul(inputX,weights01) + bias1 )
outputLayer1 = tf.nn.dropout(partOutputLayer1, keepProb)

partOutputLayer2 = tf.nn.relu( tf.matmul(outputLayer1,weights12) + bias2 )
outputLayer2 = tf.nn.dropout(partOutputLayer2, keepProb)

# our third layer is the output so we apply an output activation function
outputLayer3 = tf.sigmoid( tf.matmul(outputLayer2,weights23) + bias3 )

#one hot encodding our labels. deph = numOutputClasses
#inputYCoded = tf.one_hot( inputY,numOutputClasses )

# loss function or cost function. this function tells us the current error  !!!!!2DO +I
# L2 (Gaussian) regularization is being applied to punish the loss
loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputLayer3, labels=inputY) )  + regularizerRate*( tf.reduce_sum(tf.square(bias1)) + tf.reduce_sum(tf.square(bias2)) )
#loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=outputLayer3, labels=inputY) )  + regularizerRate*( tf.reduce_sum(tf.square(weights01)) + tf.reduce_sum(tf.square(weights12)) + tf.reduce_sum(tf.square(weights23)) )


# Variable learning rate. with this our model can start with a hight learning rate
# but will decrese to fit better our data and find the best posible values.
# every fifth epoch will decrease to 85% (15% decrease)

learningRate = tf.train.exponential_decay( starterLearningRate, 0, 5, 0.85, staircase=True )

# optimizer. This is the operator that "trains" our model (weights and biases) 
# minimizing the loss (error) given a learning rate. 2DO. Learn more abour diferent types
# Only know SGD. here we are using an adam optimizer

optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss, var_list=[weights01, weights12, weights23, bias1, bias2, bias3] )

# Metrics
correctPrediction = tf.equal( tf.argmax(inputY,1), tf.argmax(outputLayer3,1) )
accuracy = tf.reduce_mean( tf.cast( correctPrediction, tf.float32 ) )

# Now Lets train our model

#lets train in batches. 
# experiment. train with the whole data. what would happen.
batch_size = 256
epochs = 14
dropout_prob = 0.6 

training_acc = []
trainin_loss = []
test_acc = []

session.run(tf.global_variables_initializer())


for epoch in range(epochs):
    # indices to shuffle the data so each epoch is trained kind of differently
    indx = np.arange(x_train.shape[0])
    np.random.shuffle(indx)
    for index in range(0,x_train.shape[0],batch_size):
        # lets run the optimizer to train 
        session.run(optimizer , {inputX: x_train[ indx[index:index + batch_size] ], inputY: y_train[ indx[ index:index + batch_size ] ] , keepProb: 0.6} )
    
    # After training for 1 epoch lets try to predict our data
    training_acc.append( session.run( accuracy, {inputX: x_train, inputY: y_train, keepProb: 1.0 } ) )
    trainin_loss.append( session.run( loss, {inputX: x_train, inputY: y_train, keepProb: 1.0} ) )

    test_acc.append( accuracy_score( y_test, session.run(outputLayer3, {inputX: x_test, keepProb: 1.0} ).argmax(1) ) )

    print( "Epoch {0} Training Loss: {1:.3f} Training Acc: {2:.3f} Test Acc {3:.3f}".format(epoch, trainin_loss[epoch], training_acc[epoch], test_acc[epoch]) )

