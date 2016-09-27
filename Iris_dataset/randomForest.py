import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv('Iris.csv')
y = dataset.iloc[: , 5]
X = dataset.iloc[: , 0:4]

sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

        #for i in range(len(y)):
        #    if(y.iloc[i] == 'Iris-setosa'):
        #        y.iloc[i] = 1
        #    elif(y.iloc[i] == 'Iris-virginica'):
        #        y.iloc[i] = 3
        #    else:
        #        y.iloc[i] = 2

y = y.replace(to_replace= 'Iris-setosa', value = 1)
y = y.replace(to_replace= 'Iris-virginica', value = 3)
y = y.replace(to_replace = 'Iris-versicolor', value = 2)
y = y.tolist()

        #X = X.as_matrix()
        #print(type(y))
        #print(type(X))

        #print('1')
forest = RandomForestClassifier(criterion='entropy', n_estimators=10)
forest.fit(X, y)

        #print('2')

predictions = forest.predict(X)
total_correct = 0

        #print('3')

for i in range(len(y)):
    if(predictions[i] == y[i]):
        total_correct = total_correct + 1

        #print('4')

accuracy = total_correct * 1.0/ len(y)

        #print('5')

print (accuracy)
