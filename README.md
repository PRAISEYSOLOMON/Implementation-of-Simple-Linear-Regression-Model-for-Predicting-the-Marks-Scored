# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the needed packages.

2. Assigning hours to x and scores to y.
   
3. Plot the scatter plot. 

4. Use mse,rmse,mae formula to find the values.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: PRAISEY S
RegisterNumber: 212222040117
# IMPORT REQUIRED PACKAGE
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset=pd.read_csv('student_scores.csv')
print(dataset)
# READ CSV FILES
dataset=pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
# COMPARE DATASET
x=dataset.iloc[:,:-1].values
print(x)
y=dataset.iloc[:,1].values
print(y)
# PRINT PREDICTED VALUE
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
# GRAPH PLOT FOR TRAINING SET
plt.scatter(x_train,y_train,color='purple')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# GRAPH PLOT FOR TESTING SET
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
# PRINT THE ERROR
mse=mean_absolute_error(y_test,y_pred)
print('Mean Square Error = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('Mean Absolute Error = ',mae)
rmse=np.sqrt(mse)
print("Root Mean Square Error = ",rmse)
*/
```

## Output:

To Read Head and Tail Files

![no1](https://github.com/PRAISEYSOLOMON/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394259/4bd25c94-2365-4c66-bbee-66edb05777e7)

Compare Dataset

![no 2](https://github.com/PRAISEYSOLOMON/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394259/978171d5-099d-4ffe-97ca-ff7f0e73800a)

Predicted Value

![no 3](https://github.com/PRAISEYSOLOMON/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394259/1021251b-3aa2-4b8b-880b-53e6620725bf)

Graph For Training Set

![no 4](https://github.com/PRAISEYSOLOMON/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394259/2a3d7e5b-1b47-4f55-a396-83ac03a9fa28)

Graph For Testing Set

![no 5](https://github.com/PRAISEYSOLOMON/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394259/c5eefa23-607f-4803-b206-925b1cf3b6b6)

Error

![no 6](https://github.com/PRAISEYSOLOMON/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119394259/2eb83582-5ae8-4606-83e0-5f6dafaa0b39)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.

