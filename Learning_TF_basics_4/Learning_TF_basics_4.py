import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# Reading the data. This is a file with characteristics obtained from
# birds sounds using MFCC. 2DO. applying other methods of extracting characteristics

# Reading data
data = np.loadtxt('/home/felipe/nD/U de A/Machine learning/Learning_TensorFlow/Learning_TF_basics_4/DSEG2SNF.txt')
np.random.shuffle(data)

# Splitting 85% training. 15% testing
numberData = data.shape[0]
numberTrain = int(numberData*0.85)
numberTest = numberData - numberTrain
x_data = data[:,:-1]
y_data = data[:,-1]

#scaler=StandardScaler()
#scaler.fit(x_data)

x_data = (x_data-np.mean(x_data,axis=0))/(np.std(x_data,axis=0))

x_train = x_data[:numberTrain,:]
y_train = y_data[:numberTrain]

x_test = x_data[-numberTest:,:]
y_test = y_data[-numberTest:].astype(int)

lb = LabelBinarizer()

# normalizing data with z-score
#x_train = (x_train-np.mean(x_train,axis=0))/(np.std(x_train,axis=0))
#x_test = (x_test-np.mean(x_test,axis=0))/(np.std(x_test,axis=0))

y_train = lb.fit_transform(y_train)

print("Data sizes")
print("x train: {} y train: {}".format(x_train.shape, y_train.shape))
print("x test: {} y test: {}".format(x_test.shape, y_test.shape))

# Lets start with our TF model.

session = tf.Session()

# Out model.
# 52 -> 36 -> 24 -> 8. Output 8 classes.

numInputFeatures = x_train.shape[1]
numNeuronsLayer1 = 36
numNeuronsLayer2 = 20
numOutpuClasses = 8

# The rate to go down the slope
starterLearningRate = 0.01
# rate to punish the loss function.
regularizerRate=0.1

# Input data to out graph. Features, Labels, dropout_prob
inputX = tf.placeholder('float32', shape=(None,numInputFeatures)  )
inputY = tf.placeholder('float32', shape=(None,numOutpuClasses)  )
keepProb = tf.placeholder(tf.float32)

# creating our weight tensors and initializing them randomly
weights01 = tf.Variable( tf.random_normal( [numInputFeatures,numNeuronsLayer1], stddev=(1/tf.sqrt(float(numInputFeatures))) ) )
bias1 = tf.Variable( tf.random_normal( [numNeuronsLayer1] ) )

weights12 = tf.Variable( tf.random_normal( [numNeuronsLayer1,numNeuronsLayer2], stddev=(1/tf.sqrt(float(numNeuronsLayer1))) ) )
bias2 = tf.Variable( tf.random_normal( [numNeuronsLayer2] ) )

weights23 = tf.Variable( tf.random_normal( [numNeuronsLayer2,numOutpuClasses], stddev=(1/tf.sqrt(float(numNeuronsLayer2))) ) )
bias3 = tf.Variable( tf.random_normal( [numOutpuClasses] ) )

# Now lets take the previous tensors and connect them (matrix ops).
layer1 = tf.nn.relu( tf.matmul(inputX, weights01) + bias1 )
outputLayer1 = tf.nn.dropout( layer1, keepProb )

layer2 = tf.nn.relu( tf.matmul(outputLayer1, weights12) + bias2 )
outputLayer2 = tf.nn.dropout( layer2, keepProb )

# our model's output layer
prediction = tf.nn.sigmoid( tf.matmul(outputLayer2, weights23) + bias3 )


# Defining our Loss function. L2 regularization to punish the loss function
loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=inputY) ) + regularizerRate*( tf.reduce_sum(tf.square(bias1)) + tf.reduce_sum(tf.square(bias2)) + tf.reduce_sum(tf.square(bias3)) )
#loss = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=inputY) ) + regularizerRate*( tf.reduce_sum(tf.square(weights01)) + tf.reduce_sum(tf.square(weights12)) + tf.reduce_sum(tf.square(weights23)) )

# Learning rate changes. lowers 15% each epoch to ge more precise when reaching the bottom of the slope
learningRate = tf.train.exponential_decay( starterLearningRate, 0, 5, 0.85, staircase= True)

# Optimizer. Using Adam which is kind of the best ? need to learn more about diferent optimizers
optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss, var_list=[weights01, weights12, weights23, bias1, bias2, bias3])

# Metrics
correctPrediction = tf.equal( tf.argmax(inputY,1), tf.argmax(prediction, 1) )
accuracy = tf.reduce_mean( tf.cast(correctPrediction,tf.float32) )


# train
batch_size = 128
epochs = 100
dropout_prob = 0.85

train_acc = []
train_loss = []
test_acc = []

session.run( tf.global_variables_initializer() )

for epoch in range(epochs):
    toshuffle = np.arange(x_train.shape[0])
    np.random.shuffle(toshuffle)
    for index in range(0,x_train.shape[0], batch_size):
        session.run(optimizer, {inputX: x_train[toshuffle[index:index+batch_size]],inputY: y_train[toshuffle[index:index+batch_size]], keepProb: dropout_prob})
    #session.run(optimizer, {inputX: x_train, inputY: y_train, keepProb: dropout_prob})
    train_acc.append( session.run(accuracy, {inputX: x_train,inputY: y_train, keepProb: 1.0}) )
    train_loss.append( session.run(loss, {inputX: x_train,inputY: y_train, keepProb: 1.0}) )
    #print(session.run(prediction, {inputX: x_test, keepProb: 1.0})[0])
    test_acc.append( accuracy_score( y_test, session.run(prediction, {inputX: x_test, keepProb: 1.0}).argmax(1) +1 ) )

    print( "Epoch {0} Training Loss: {1:.3f} Training Acc: {2:.3f} Test Acc {3:.3f}".format(epoch, train_loss[epoch], train_acc[epoch], test_acc[epoch]) )