import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
import time


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

print('Data Set up!. Starting Trees')

##########Starting Tree Processing
list_of_accuracy = []
list_of_time = []

for forestrun in range(10):
#preprocessing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    stdscX = StandardScaler()
    X_train_std = stdscX.fit_transform(X_train)
    X_test_std = stdscX.transform(X_test)



#building a tree
    start_time = time.time()
    forest = RandomForestClassifier(criterion = 'entropy', n_estimators = 10, n_jobs = 2)
    forest.fit(X_train_std,y_train)
    predictions = forest.predict(X_test_std)

    sum = 0;
    for i in range(len(y_test)):
        if (predictions[i] == y_test[i]):
            sum = sum + 1;

    list_of_accuracy.append(sum * 1.0 / len(y_test));
    list_of_time.append(time.time() - start_time)

print (list_of_accuracy);
print (list_of_time);
