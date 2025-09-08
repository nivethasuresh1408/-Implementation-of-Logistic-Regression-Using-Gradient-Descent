# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import libraries and load the dataset.
2.Preprocess data and define X (features) and y (target).
3.Add bias term and initialize weights θ.
4.Apply gradient descent using sigmoid to update θ.
5.Predict outcomes and evaluate accuracy.


## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Nivetha S
RegisterNumber: 212223040137 
*/
```
```
import pandas as pd
import numpy as np
data = pd.read_csv("Placement_Data.csv")
data1 = data.drop(['sl_no','salary'], axis=1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in ["gender","ssc_b","hsc_b","hsc_s","degree_t","workex","specialisation","status"]:
    data1[col] = le.fit_transform(data1[col])
X = data1.iloc[:,:-1].values
y = data1["status"].values
X = np.c_[np.ones((X.shape[0], 1)), X]
theta = np.random.randn(X.shape[1])
def sigmoid(z):
    return 1 / (1 + np.exp(-z))
def loss(theta, X, y):
    h = sigmoid(X.dot(theta))
    return -np.sum(y*np.log(h) + (1-y)*np.log(1-h))
def gradient_descent(theta, X, y, alpha, num_iterations):
    m = len(y)
    for i in range(num_iterations):
        h = sigmoid(X.dot(theta))
        gradient = X.T.dot(h - y) / m
        theta -= alpha * gradient
    return theta
theta = gradient_descent(theta, X, y, alpha=0.01, num_iterations=1000)
def predict(theta, X):
    h = sigmoid(X.dot(theta))
    y_pred = np.where(h >= 0.5, 1, 0)
    return y_pred
y_pred = predict(theta, X)
accuracy = np.mean(y_pred.flatten() == y)
print("Accuracy:", accuracy)
print("Predicted:\n",y_pred)
print("Actual:\n",y)
xnew = np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
xnew = np.insert(xnew, 0, 1, axis=1) 
y_prednew = predict(theta, xnew)
print('Name: Nivetha S')
print('Reg No: 212223040137')
print("Predicted Result:", y_prednew)
```

## Output:

<img width="1384" height="471" alt="image" src="https://github.com/user-attachments/assets/de68010f-d790-40e3-80b5-99afce853b66" />

<img width="1424" height="237" alt="image" src="https://github.com/user-attachments/assets/1e3a2fbe-e09a-411b-80b5-756f90d074af" />


## Result:

Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

