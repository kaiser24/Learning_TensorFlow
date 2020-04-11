from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

data = np.loadtxt('DSEG2SNF.txt')
np.random.shuffle(data)
numberData = data.shape[0]
numberTrain = int(numberData*0.85)
numberTest = numberData - numberTrain
x_data = data[:,:-1]
y_data = data[:,-1]

#scaler=StandardScaler()
#scaler.fit(x_data)
#x_data = scaler.transform(x_data)
#x_data = (x_data-np.mean(x_data,axis=0))/(np.std(x_data,axis=0))

x_train = x_data[:numberTrain,:]
y_train = y_data[:numberTrain]

x_test = x_data[-numberTest:,:]
y_test = y_data[-numberTest:].astype(int)

KNear=KNeighborsClassifier(n_neighbors=8)
KNear.fit(x_train,y_train)

train_pred = KNear.predict(x_train)
test_pred = KNear.predict(x_test)
print('Training acc',accuracy_score(train_pred,y_train))
print('Training acc',accuracy_score(test_pred,y_test))