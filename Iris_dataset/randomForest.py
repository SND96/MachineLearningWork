import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

dataset = pd.read_csv('Iris.csv')
y = dataset.iloc[: , 5]
X = dataset.iloc[: , 0:4]

sc = StandardScaler()
sc.fit(X)
X = sc.transform(X)

y = y.replace(to_replace= 'Iris-setosa', value = 1)
y = y.replace(to_replace= 'Iris-virginica', value = 3)
y = y.replace(to_replace = 'Iris-versicolor', value = 2)
y = y.tolist()

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3)

forest = RandomForestClassifier(criterion='entropy', n_estimators=10)
forest.fit(X_train, y_train)

predictions = forest.predict(X_test)
total_correct = 0

for i in range(len(y_test)):
    if(predictions[i] == y_test[i]):
        total_correct = total_correct + 1

accuracy = total_correct * 1.0/ len(y_test)

print (accuracy)
