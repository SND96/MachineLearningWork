# HT_SensorData_Tree
Link To dataset - http://archive.ics.uci.edu/ml/datasets/Gas+sensors+for+home+activity+monitoring

This Code  tries to predict  the first column (id_nos)  using all the other attributes. I have replaced the corresponding  id_nos with class  labels from metadata.

Based on the paper:
Ramon Huerta, Thiago Mosqueiro, Jordi Fonollosa, Nikolai Rulkov, Irene Rodriguez-Lujan. Online Decorrelation of Humidity and Temperature in Chemical Sensors for Continuous Monitoring. Chemometrics and Intelligent Laboratory Systems 2016.

The script Cross_validation_on_set.py runs a 5-fold Cross Validation run on the set.
The first set of runs are using attributes (R1,R2,R3,R4,R5,R6,R7,R8).
The next set of runs use attributes (R1,R2,R3,R4,R5,R6,R7,R8,Temp,Humidity).
