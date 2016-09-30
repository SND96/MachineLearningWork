import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
#IMPORTS COMPLETE

#opening Sensor Data
with open('HT_Sensor_dataset.dat', 'r') as f:
    next(f)  #skip first row
    df = pd.DataFrame(l.rstrip().split() for l in f)

#opening metadata
metadata = np.loadtxt('HT_Sensor_metadata.dat', skiprows=1, dtype=str)
metadata[ metadata[:,2] == "b'wine'", 2 ] = 1
metadata[ metadata[:,2] == "b'banana'", 2 ] = 2
metadata[ metadata[:,2] == "b'background'", 2 ] = 0

#changing labels in metadata
id_nos = metadata[:,2];
id_nos_new = np.empty([len(id_nos),1])

for i in range(len(id_nos)):
    id_nos_new[i] = np.int(id_nos[i])

id_nos = id_nos_new

df = df.as_matrix();
X = df[:,1:];
y = df[:,0];

y_new = np.empty(y.shape)
for i in range(len(y)):
    y_new[i] = id_nos[ np.int(y[i]) ]
y = y_new;

stdscX = StandardScaler()
tree = DecisionTreeClassifier(criterion='entropy')

####################################################
################ DATA - SET - UP ###################
####################################################

##### Run 1 ###############
##### Running on RS #######
print("\t\tRunning on RS Feature Set")
XRS = X[:,1:8];
XRS_std = stdscX.fit_transform(XRS)

X_train,X_test,y_train,y_test = train_test_split(XRS_std , y , test_size = 0.2)

kf = KFold(len(X_train),5, shuffle = True)

for k, (train_index, test_index) in enumerate(kf):

    #Setting variables so Code is more readable
    X_train_CV = X_train[train_index]
    X_test_CV = X_train[test_index]
    y_train_CV = y_train[train_index]
    y_test_CV = y_train[test_index]
    #Fitting the Tree
    tree.fit(X_train_CV,y_train_CV)

    #For Cross Validation Set
    predictions = tree.predict(X_test_CV)
    sum = 0;
    for i in range(len(predictions)):
        if (predictions[i] == y_test_CV[i]):
            sum = sum + 1;
    CV_accuracy = sum*1.0/(len(y_test_CV))

    #For Test Set
    predictions = tree.predict(X_test)
    sum = 0
    for i in range(len(predictions)):
        if(predictions[i] == y_test[i]):
            sum = sum+1
    Test_accuracy = sum * 1.0 / (len(y_test))

    print("\tRUN ", k, ": CV Accuracy = ", CV_accuracy, "\t Test Accuracy = ", Test_accuracy)



  ########End of Previous Run##########

  ########Running it on RS, T, H ######
print("\t\tRunning on RSTH Feature Set")
XRSTH = X[:,1:];
XRSTH_std = stdscX.fit_transform(XRSTH)

X_train,X_test,y_train,y_test = train_test_split(XRSTH_std , y , test_size = 0.2)

kf = KFold(len(X_train),5, shuffle = True)

for k, (train_index, test_index) in enumerate(kf):

    #Setting variables so Code is more readable
    X_train_CV = X_train[train_index]
    X_test_CV = X_train[test_index]
    y_train_CV = y_train[train_index]
    y_test_CV = y_train[test_index]
    #Fitting the Tree
    tree.fit(X_train_CV,y_train_CV)

    #For Cross Validation Set
    predictions = tree.predict(X_test_CV)
    sum = 0;
    for i in range(len(predictions)):
        if (predictions[i] == y_test_CV[i]):
            sum = sum + 1;
    CV_accuracy = sum*1.0/(len(y_test_CV))

    #For Test Set
    predictions = tree.predict(X_test)
    sum = 0
    for i in range(len(predictions)):
        if(predictions[i] == y_test[i]):
            sum = sum+1
    Test_accuracy = sum * 1.0 / (len(y_test))

    print("\tRUN ", k, ": CV Accuracy = ", CV_accuracy, "\t Test Accuracy = ", Test_accuracy)
