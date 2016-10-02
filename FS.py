import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation
from sklearn.cross_validation import KFold
import time
from math import exp
beta = np.array([
[-0.0044,		0.00014, 	0.0110],
[-0.0110,		0.00034, 	0.0240],
[-0.0110,		0.00034,	0.0230],
[-0.0110,		0.00033,	0.0230],
[-0.0056, 		0.00018,	0.0086],
[-0.0039,		0.00012,	0.0071],
[-0.0070,		0.00022,	0.0095],
[-0.0057,		0.00020,	0.0029]
])
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

(rows,cols) = X.shape
X_fs = np.empty([rows-100,cols])

for i in range(8):
	diff_H = X[i][9] - X[i-1][9]
	diff_T = X[i][10] - X[i-1][10]
	for j in range(len(X)):
		if (id_no[j] != id_no[j-1]):
			continue
		X_fs[i][j] = X[i][j] - X[i][j-1]*exp( beta[i][0]*(diff_H) + beta[i][1]*(diff_H)*(diff_H) + beta[i][2]*(diff_H)*(diff_T))
